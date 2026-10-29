---
name: loom
description: Entry point for AMD Loom work (hrx-loom SDK, loom-serve, loom-fetch, Loom IR kernels) on AMD GPUs/NPUs such as Strix Halo — installing, serving catalog models, the demo web UI, porting a new LLM to Loom, optimizing it while preserving correctness, and benchmarking against llama.cpp. Use whenever the user mentions Loom, loom-serve, loom-fetch, hrx-loom, .loom files, or evaluating/porting models for Loom.
---

# Loom (umbrella)

Loom = AMD's JIT compiler + runtime + model server (`hrx-loom` on PyPI, 0.1.0 pinned by these skills; source at
github.com/ROCm/hrx-system, models at github.com/ROCm/hrx-loom-library). Models
ship as **Loom IR source** (`.loom` kernels + command/control programs) plus a
package card (`model.json`) that points at unmodified checkpoints (GGUF or
safetensors). `loom-serve` compiles the program for the live device at startup
and exposes OpenAI-style HTTP endpoints.

## Pick the sub-skill

| Task | Skill |
|---|---|
| Install the SDK + library, fetch a catalog model, start/stop `loom-serve`, NPU serving | `loom-setup` |
| Browser playground (chat / ASR / TTS / image / embeddings) over running servers | `loom-demo-ui` |
| Make an arbitrary GGUF LLM servable by loom-serve (write the program) | `loom-port-llm` |
| What is possible for this model? Pick optimization targets (roofline vs llama.cpp) | `loom-roofline` (`${CLAUDE_SKILL_DIR}/../loom-roofline/SKILL.md`) |
| Make a ported model faster without breaking correctness | `loom-optimize` |
| Correctness gate vs llama.cpp, throughput methodology, reporting | `loom-benchmark` |

All paths in these skills assume the install root `~/loom` (SDK venv `~/loom/.venv`,
tools venv `~/loom/.tools`). Substitute your own root consistently. The scripts take overrides:
`VENV=` (serve.sh, run_kernel_tests.sh), `UI_VENV=` or the 2nd argument (start_ui.sh),
and `LOOM_SERVE_STATE=` (the serve.sh PID/log dir).

## Windows

The whole pipeline (install, port, kernel tests, golden gate, perf, serve) works on
Windows 11 on Strix Halo. Run the scripts from **Git Bash**. Differences:
- Venv executables are in `Scripts/`, not `bin/`. Read every `~/loom/.venv/bin/X`
  and `~/loom/.tools/bin/python` in these skills as `~/loom/.venv/Scripts/X` and
  `~/loom/.tools/Scripts/python`. The scripts detect this themselves.
- The SDK is the same PyPI release (`hrx_loom-0.1.0-py3-none-win_amd64.whl`); it needs the
  Microsoft Visual C++ v14 x64 redistributable. `--device=amdgpu` reports `driver: amd,
  backend: libamdf` in `/v1/status`.
- Python: `install_loom.sh` uses `py -3` (override with `PYTHON=`). On Python < 3.15,
  text files default to cp1252, so open text files with `encoding="utf-8"`.
- `ss`, `pgrep` and `/sys` don't exist. Use these instead:
  - ports: `netstat -ano -p tcp | grep LISTENING`;
  - processes: `ps -W | grep -E 'loom-serve|llama-server'`;
  - GPU memory: `llama-server --list-devices` shows the HIP-visible VRAM (91.6 GiB
    on a 128 GB box with a 32 GiB BIOS carve-out).
- `build.py` symlinks the GGUF. Without Developer Mode it hard-links it instead,
  which needs the package and the GGUF on the same drive.
- **llama.cpp:** lemonade installs builds in `~/.cache/lemonade/bin/llamacpp/{rocm,rocm-stable,rocm-preview}`.
  A build missing its HIP/BLAS DLLs **silently runs on CPU**: on the original
  machine only one of the three saw the GPU. Check with `llama-server --list-devices`.
  `llamacpp_server.sh` refuses a build that lists no `ROCm` device, so
  `pick_llamacpp.sh` skips it.

Typical cold-start order: `loom-setup` → (`loom-demo-ui`) → `loom-port-llm` →
`loom-benchmark` (baseline + golden) → `loom-roofline` (targets) → `loom-optimize` ↔ `loom-benchmark`.

**"Optimize this model".** Run `loom-roofline`. It turns the request into a goal
and a time budget:
- with no target and no time given, the goal is the roofline targets and the
  budget is **1 h** of the user's time;
- a stated target and/or a time budget replaces the defaults.

Announce the targets and the budget without asking, then iterate in
`loom-optimize`. After each measurement, `progress.py` checks how fast the gap to
the roofline is closing, per hour, and gives one of four verdicts:
- continue;
- review (pivot to another path, or fork from an earlier decision);
- budget used up;
- done.

Report met or short for each metric, with the approach curve.

## End-to-end requests: one turn

"Get X working, optimize it and report" is a single job. Run the whole pipeline
without stopping:
1. reuse the install;
2. port (native context, tools: see `loom-port-llm`);
3. pass the gates: kernel tests, golden, `agent_golden.py` (tool loop) and `needle.py` (full context);
4. measure the llama.cpp baseline;
5. run `loom-roofline` to set targets and a budget. This is **not optional**: the
   report must contain the roofline table (`bench/roofline.json`) and the approach
   curve (`bench/targets.jsonl`), and a report without them is incomplete;
6. optimize, under the rate controller;
7. pass the gates again;
8. serve;
9. run `loom-benchmark/scripts/audit.py` from `bench/`. It must print `AUDIT PASS`
   (fix any FAIL first). Then report, using its headline table.

Stop to ask only for what the hard rules require: a gated-license click, or
approval of a binary you downloaded. Avoid both where possible:
- **Reuse what is installed.** Check `~/loom/.venv/bin/loom-serve` (Windows: `.venv/Scripts/loom-serve.exe`) and the HF cache.
- **Baseline = the fastest llama.cpp ROCm build already on the machine.** Run
  `loom-benchmark/scripts/pick_llamacpp.sh MODEL.gguf`. It finds and benchmarks
  every HIP build. Version strings can't be trusted, and an old build inflated a
  margin from +25 % to 2.5× in one run. A build the user already installed doesn't
  need approval; one you download in this session does.

The `loom-roofline` skill is at `${CLAUDE_SKILL_DIR}/../loom-roofline/SKILL.md`.
Read it from there if it isn't in your skill list.

## Hard rules (learned the expensive way)

- **Never load a model whose weights exceed the GPU's GTT/unified-memory budget.**
  On the Strix Halo box (128 GB RAM) the GPU limit is ~61 GiB GTT. Qwen3.8-Flash-Next
  (104 GiB) hard-crashed the machine during the original evaluation. Never retry it on a 128 GB box. Check
  `du -shL <weights>` against `cat /sys/class/drm/card*/device/mem_info_gtt_total`
  before serving anything big.
- **Always pass `--memory_bytes`** to `loom-serve` (weights + KV pool + ~2 GiB
  workspace). Too small → HTTP 503 "retained state leaves insufficient completion
  capacity". Several models can share one budget with `--parameter_idle_ms=60000`.
- **Freshly downloaded binaries need explicit user approval before running**
  (e.g. llama.cpp release zips). Loom itself is installed from PyPI with pip.
- Gated Hugging Face models: give the user the exact model page URL so they can
  accept the license; don't try to work around the gate.
- Every chat/completions POST needs `Content-Type: application/json`. (Python
  `requests` with an empty `Expect:` header breaks the connection.)
- The first request after startup triggers lazy weight loading (seconds to a
  minute); don't time it.
- Stop servers by PID file (`loom-setup/scripts/stop.sh`). `pkill -f "...port..."`
  can match and kill your own shell.
- After a rebuild, make sure the old server is really gone. `serve.sh` refuses a
  busy port and checks that its own PID owns the port before printing READY.
  READY after about 1 s is normal, since compiled kernels are cached and weights
  load on the first request. A stale server shows up as a READY from a PID that
  isn't yours, or as old behavior.
- Before starting llama-server or loom-serve, check the port is free (`ss -ltnp`; Windows: `netstat -ano`).
- Golden references must be external (llama.cpp ROCm or the model's own reference
  implementation), never a numpy model you wrote yourself.

## Facts about this hardware (Strix Halo, Ryzen AI Max+ 395)

- GPU gfx1151 (RDNA 3.5), wave32, WMMA f16→f32 (~59 TFLOP/s peak), ~216 GB/s
  measured DRAM bandwidth, 32 MB MALL (last-level cache), ~61 GiB GTT.
- NPU: XDNA (`--device='xdna://0?columns=8' --pool_backing=fixed`). Only some
  catalog models (e.g. Qwen3-Embedding-0.6B) run there, and it needed an older
  SDK wheel (see `loom-setup`).
- `loom-benchmark` hardware counters are UNIMPLEMENTED on this GPU. Profile by
  ablation (remove or double a kernel, measure end to end) instead.

## Where things are documented upstream

- The docs site is rocm.github.io/hrx-system/loom. The same docs are in the repo:
  `git clone --depth 1 https://github.com/ROCm/hrx-system` and open
  `loom/docs/src/`.
  - `guide/` covers the IR: values, buffers/views, kernels, vectors, facts,
    and command programs.
  - `serving/model-programs.md` covers package authoring.
  - `serving/workflows/fetch-and-serve.md` and `live-model-development.md`
    cover the serving workflows.
  - `workflows/test-correctness.md` covers `check.case` tests.
- Each tool prints an agent-oriented manual: `loom-test --agents_md`,
  `loom-compile --agents_md`, `loom-serve --agents_md`, `loom-opt --agents_md`,
  `loom-run --agents_md`, `loom-format --agents_md`.
- Library model programs are the best worked examples. For example,
  `library/program/text/qwen3_8/chat.loom` is a ChatML control program with
  `<think>` parsing, and `library/model/*/` holds the cards.
