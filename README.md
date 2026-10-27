# Loom skills for Claude Code

[Claude Code](https://claude.com/claude-code) skills that let a fresh session use
[AMD Loom](https://github.com/ROCm/hrx-system) (the `hrx-loom` compiler, runtime and
`loom-serve` model server) from a cold start:

- install the SDK;
- fetch and serve catalog models (LLM, ASR, TTS, image, embeddings);
- run a browser playground;
- **port an arbitrary GGUF LLM to Loom**;
- optimize it without breaking correctness;
- benchmark it against llama.cpp ROCm.

The skills were developed and verified on an AMD Ryzen AI Max+ 395 "Strix Halo"
(gfx1151 GPU + XDNA NPU, 128 GB).

## Install

**As a Claude Code plugin (recommended).** Run these inside Claude Code:

```
/plugin marketplace add lemonade-sdk/loom
/plugin install loom@loom
```

Or from a shell:

```bash
claude plugin marketplace add lemonade-sdk/loom
claude plugin install loom@loom
```

Then start a new session. To update later, run
`claude plugin marketplace update loom && claude plugin update loom@loom`.

> While this repo is private, you need read access to it and working git credentials
> for github.com, because Claude Code clones it with your existing git setup:
> `gh auth login && gh auth setup-git`, or an SSH key loaded in ssh-agent.

**Or as plain user skills**, by copying them into `~/.claude/skills`:

```bash
git clone https://github.com/lemonade-sdk/loom && ./loom/install.sh      # ./install.sh --uninstall to remove
```

## Use

### Recommended: port, optimize and report in one prompt

```
/loom get <hf-org>/<model> working on loom-serve, optimize it, and report the performance relative to llamacpp rocm
```

Claude runs the whole pipeline without stopping:
1. reuse or install Loom;
2. port the GGUF;
3. pass the token-by-token correctness gate against llama.cpp ROCm;
4. measure the llama.cpp baseline;
5. run a **roofline analysis** that sets realistic per-metric targets;
6. optimize under a rate-of-approach controller;
7. re-run the gates;
8. serve the model and report.

It stops to ask only for a gated-model license or approval of a downloaded binary.

Tested on nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB (Gemma-4 E4B architecture), with a
fresh agent and this single prompt. The run took 79 minutes. The results, against
llama.cpp ROCm b1355 on the same GGUF:

| Metric | Margin |
|---|---|
| Decode | +14 % |
| Prefill 1k | +38 % |
| Prefill 2k | +39 % |
| 4 streams | +7 % |

Every divergence from llama.cpp's output was a near-tie, up to 32k-token prompts.

**Effort control.** With no target and no time given, the optimization phase gets
**1 hour**. Add either to the prompt to change it, for example "…optimize it for 3
hours…" or "…until decode is 50 % faster than llama.cpp…". The roofline step checks
a stated target against the hardware ceiling and says so up front if it is out of
reach.

### Other examples

Ask in plain language, for example:
- "install Loom and serve moonshine";
- "port Qwen3-4B-Instruct-2507 to loom-serve and validate it against llama.cpp";
- "benchmark my Loom port";
- "optimize this model" (roofline → targets → about 1 h of optimization → report);
- "what's the roofline for this model?" or "is +50 % decode achievable?".

The skill descriptions make Claude pick the right one. You can also invoke a skill by
name:

| Plugin install | Copy install | What it does |
|---|---|---|
| `/loom:loom` | `/loom` | Start here: overview, hard safety rules, which skill to use |
| `/loom:loom-setup` | `/loom-setup` | Install the SDK wheel and the model library, `loom-fetch`, start/stop/inspect `loom-serve` (GPU and NPU) |
| `/loom:loom-demo-ui` | `/loom-demo-ui` | Web playground: chat, transcribe, speak, image, embed. It finds models from the servers' `/v1/status` |
| `/loom:loom-port-llm` | `/loom-port-llm` | Config-driven starter kit that turns a dense GGUF LLM into a Loom package: inspect → generate kernels → build → kernel unit tests → golden check |
| `/loom:loom-roofline` | `/loom-roofline` | Roofline for decode/prefill/c4 from the GGUF, calibrated against achieved Loom efficiency; picks per-metric targets vs llama.cpp and a time budget (default 1 h); a rate-of-approach controller decides continue / pivot / stop |
| `/loom:loom-optimize` | `/loom-optimize` | The measure / ablate / change / verify loop, a kernel A/B helper, a catalog of optimizations that worked (and that didn't), and the sources of a hand-tuned MoE port |
| `/loom:loom-benchmark` | `/loom-benchmark` | Token-by-token golden comparison against llama.cpp ROCm, a batching-consistency check, throughput workloads, reporting rules |

A full cold-start port takes about 10–25 minutes for a small dense model, and about
35 minutes for an architecture the starter kit doesn't cover (Gemma-4 E4B). In
order:
1. install (≈15 s);
2. port a model;
3. pass the golden check;
4. measure performance against llama.cpp;
5. roofline and targets;
6. optimize (1 h by default).

## Requirements

- Linux, and an AMD GPU (ROCm-class; developed on gfx1151). The NPU path needs an
  XDNA device.
- `gh`, authenticated: the Loom SDK wheel is a GitHub Actions artifact of
  ROCm/hrx-system, not a PyPI package.
- Python ≥ 3.12 with `venv`, plus `git` and `unzip`.
- For the golden reference: a llama.cpp ROCm build from
  [lemonade-sdk/llamacpp-rocm](https://github.com/lemonade-sdk/llamacpp-rocm). The
  skill asks before running any downloaded binary.

## Safety notes baked into the skills

- **Memory limit.** Never serve models whose weights exceed the GPU's memory budget:
  a 104 GiB model hard-crashed a 128 GB machine.
- **`--memory_bytes`.** Always pass it to `loom-serve`.
- **Downloaded binaries.** Ask before running one.
- **Golden references.** They are external (llama.cpp), never a model written by
  hand.

## Repository layout

```
.claude-plugin/{plugin.json,marketplace.json}   # plugin + single-plugin marketplace
skills/<skill>/SKILL.md                          # the six skills, with their scripts and references
install.sh                                       # copy-install alternative
LICENSE                                          # MIT
THIRD_PARTY_NOTICES.md
```

Skills refer to their bundled files through `${CLAUDE_SKILL_DIR}`, so the same tree
works both as a plugin and when copied to `~/.claude/skills`.

## Contributing

To try local changes without installing, run `claude --plugin-dir /path/to/loom`, and
check the manifests with `claude plugin validate .`. When you fix a skill, rerun a
cold-start test: a fresh subagent follows the skills literally in a new install root
and reports every wrong or missing instruction. That is how these skills were
debugged.

## License

MIT; see [LICENSE](LICENSE). Bundled third-party material keeps its own license; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
