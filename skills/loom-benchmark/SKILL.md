---
name: loom-benchmark
description: Correctness and performance methodology for Loom models vs llama.cpp ROCm — golden token-by-token comparison with log-prob near-tie analysis, a multi-turn tool-calling (agent) gate, a full-context needle gate, batching consistency, decode/prefill/long-context/concurrency throughput on real-text workloads, and how to report results honestly (tables, executive artifact). Use for "benchmark", "compare with llama.cpp", "is the port correct", "performance report".
---

# Benchmark and verify Loom models

Scripts are in `${CLAUDE_SKILL_DIR}/scripts/`; run them with the tools
venv (`~/loom/.tools/bin/python`). Workload documents are in `data/doc_{1000,2000,8000,16000,32000}.txt`.
They are real text (Loom's own markdown docs) and their sizes are approximate token counts.
Scripts that take `PACKAGE` read `llm_config.json`, `tokenizer.json` and the model
id from a package built by `loom-port-llm/starter/build.py`.

## 0. Reference: llama.cpp ROCm (not Vulkan, not CPU)

**First, use what is already on the machine, and pick the fastest.**
```bash
S=${CLAUDE_SKILL_DIR}/scripts
$S/pick_llamacpp.sh MODEL.gguf     # finds every llama-server with libggml-hip.so, benchmarks each, ranks them
```
The baseline is the top-ranked build. Don't choose by version string: the lemonade
nightly prints `version: 1`, and the tag is usually only in the directory name.
Builds differ a lot on the same GGUF:
- LFM2.5-24B prefill: b1274 1,062 tok/s vs b1355 2,140 tok/s;
- Bolt-E4B with 4 streams: b1274 125 tok/s vs b1355 157 tok/s.

A weaker baseline inflates every margin you report. Report the chosen build's path
and tag.

If none is installed, download the lemonade-sdk ROCm build:
```bash
mkdir -p ~/llamacpp-rocm && cd ~/llamacpp-rocm
gh release download -R lemonade-sdk/llamacpp-rocm -p 'llama-*-ubuntu-rocm-gfx1151-x64.zip'   # latest; or pin a tag, e.g. b1355
unzip -q llama-*-gfx1151-x64.zip && chmod +x llama-server
```
On Windows, download the `llama-*-windows-rocm-gfx1151-x64.zip` asset instead, or reuse a
lemonade build from `~/.cache/lemonade/bin/llamacpp/`. Either way, check that
`llama-server --list-devices` shows `ROCm0`, because a build missing its DLLs falls back to CPU silently.
**Ask the user before running a freshly downloaded binary.** Pick the zip that matches the GPU (`gfx1151` = Strix
Halo). Then:
```bash
LLAMACPP_DIR=<dir> $S/llamacpp_server.sh MODEL.gguf 8130 1 34000   # goldens / long context (1 slot, 34k)
LLAMACPP_DIR=<dir> $S/llamacpp_server.sh MODEL.gguf 8130 4 8192    # perf incl. 4 streams
kill $(cat /tmp/llamacpp-8130.pid)
```
Always use the **same GGUF file** as the Loom package. Never run llama.cpp and a
big Loom model at the same time if together they exceed GPU memory.

## 1. Correctness gates (before any performance claim)

```bash
S=${CLAUDE_SKILL_DIR}/scripts; P=~/work/mymodel/pkg; T=~/loom/.tools/bin/python; mkdir -p ~/work/mymodel/bench && cd ~/work/mymodel/bench   # outputs land here, not in the skill
$T $S/golden.py $P golden.json --llama http://127.0.0.1:8130 --long 1000,2000,16000,32000    # once per model/GGUF
$T $S/golden.py $P golden2.json --llama http://127.0.0.1:8130 --long 1000,2000,16000,32000   # 2nd recording: llama.cpp is not run-to-run deterministic
$T $S/compare.py $P golden.json --alt golden2.json --url http://127.0.0.1:8160   # -> per-prompt lines + {"verdict": "PASS"|"FAIL", ...}
$T $S/consistency.py http://127.0.0.1:8160 --model <model id>         # concurrent == sequential
$T $S/agent_golden.py $P http://127.0.0.1:8160 --llama http://127.0.0.1:8130   # tool loop: rendering, reuse, calls, golden per turn
$T $S/needle.py $P http://127.0.0.1:8160 --llama http://127.0.0.1:8130         # retrieval at 25-100% of context_capacity + over-limit 400
$T $S/sampling_check.py $P http://127.0.0.1:8160 --llama http://127.0.0.1:8130 # temperature/top_k/top_p/seed semantics + first-token distribution
```
For `needle.py` sizes above the llama.cpp slot's context, drop `--llama` (Loom-only:
it then must find the needle) or give `--sizes`. A 262k-token prefill takes minutes.
- **How golden.py works.** It renders each prompt with the package's chat block
  and sends the *token ids* to llama.cpp `/completion` (greedy, `n_probs: 5`). Both
  engines therefore see identical prompts: 20 short prompts × 96 tokens, plus 2
  prompts per long document × 64 tokens.
- **How compare.py works.** It asks Loom for the same completion through chat,
  rebuilds the raw text from the reasoning/answer channels, re-tokenizes it, and
  finds the first divergence. It reports the **golden log-prob gap**
  logp(golden) − logp(loom token) at that point.
- **Interpretation.** Greedy decoding through different summation orders
  *always* diverges eventually at near-ties.
  - **PASS**: every divergence has gap ≤ 1.0 and the token is in the golden top-5.
    Typical healthy ports have max gap 0.1–0.6, and full matches on a quarter or
    more of the prompts.
  - **FAIL**: a large gap, or a token outside the top-5. That is a real bug, even
    if the text reads fine.
  - Two llama.cpp builds (Vulkan vs ROCm) disagree with each other in the same
    near-tie way, which is useful context for readers.
- llama.cpp ROCm itself changes greedy output between identical runs at near-ties
  (seen at 1k and 16k on Qwen3-4B). That is why there are two recordings: `--alt`
  scores each prompt against whichever recording Loom follows further. Before
  declaring FAIL on a single prompt, re-record it; a gap that disappears was noise.
- Golden ids come from llama.cpp's `tokens`. `completion_probabilities` omits
  byte-fallback tokens (emoji, partial UTF-8), so top-5 lists are kept only while
  they line up.
- Document sizes are approximate. With a 262k-vocab tokenizer (Gemma),
  `doc_32000` is about 35k tokens, which is over a 32768 context. Either build the
  package (and llama-server) with a 40960 context, or use `--long 1000,2000,16000`
  and say so.
- Long goldens (16k/32k) are what catch KV paging, prefill chunking and attention
  masking bugs.
- `consistency.py` catches mixed-sequence epoch bugs (padding, tile uniformity,
  fallback lists). If it fails, find the first divergent token and its golden
  log-prob gap. A divergence is acceptable only when **both** hold:
  - it is a near-tie (gap < 0.05);
  - the cause is identified and stated in the report. For example, decode tokens
    that share an epoch with a prefill go through the f16 GEMM path instead of the
    f32 matvec path.

  Anything else is a bug.
- 503 errors on long prompts mean `--memory_bytes` is too small for the KV pool,
  not a model bug.
- **agent_golden.py** runs three tool-using tasks (weather lookups, a calculator)
  with deterministic tool results, greedy, and checks every assistant turn:
  - `render_ok`: Loom's `/v1/chat/completions/input_tokens` equals the token count
    of the template-exact rendering (`chat_probe.simulate`, verified against the
    Jinja template). A mismatch is a control-program bug.
  - `reuse_ok`: inside a tool loop, `cached_tokens` covers ≥ 80 % of the previous
    prompt + completion (history reuse works, including recurrent state).
  - `calls_ok`: tool calls name a declared function with JSON arguments.
  - golden: Loom's turn vs llama.cpp `/completion` on the same prompt ids; a
    divergence must be a near-tie (gap < 0.5).
  "The model ended inside its reasoning" notes are a greedy-decoding behavior of
  small thinking models, confirmed when llama.cpp's own distribution at that
  point prefers EOS; not a Loom bug by itself.
- **sampling_check.py**: temperature 0 and `top_k` 1 equal greedy, a fixed seed
  reproduces, different seeds vary, and 200 temperature-1 draws of the first token
  match llama.cpp's probabilities on the same ids (total-variation distance within
  sampling noise; Qwen3-0.6B: 0.011).
- **needle.py** hides an access code in a real-text haystack at 25/50/75/100 % of
  the package's `context_capacity` and asks for it (thinking off when the template
  allows), then sends a prompt over the capacity (must be a clean HTTP 400, server
  still healthy). A size passes when Loom finds the needle, or with `--llama` at
  least as often as llama.cpp on the same ids: small models miss needles on their
  own. Divergences from llama.cpp grow slowly with length (gaps 0.4–0.7 nats at
  30–40k on Qwen3-0.6B, exact matches in between); a step at a power of two (32768,
  65536) would be a bug.

## 2. Performance

```bash
$T $S/perf.py http://127.0.0.1:8160 loom-baseline --model <id> --log perf_log.jsonl
$T $S/perf.py http://127.0.0.1:8160 loom-baseline --model <id> --long --log perf_log.jsonl
$T $S/perf.py http://127.0.0.1:8130 llama-rocm --kind llama --log perf_log.jsonl     (and --long, with the 1-slot 34k server)
$T $S/c4_steady.py http://127.0.0.1:8160 4 --model <id>;  $T $S/c4_steady.py http://127.0.0.1:8130 4
```

| Metric | Definition |
|---|---|
| decode_tps | 300-token essay from a short prompt. Client-side: (tokens−1)/(total−TTFT). |
| prefill[] | 1k and 2k-token documents with max_tokens=4. prompt_tokens / TTFT, the median of 5 runs (`--repeats`); single runs varied 1.5–2.9k tok/s on one build. |
| decode_tps_1k / _2k / _32k | Decode with that much context in the KV cache. |
| prefill_long[] | 8k / 16k / 32k documents. |
| c4_agg_tps | 8 requests through 4 threads, total tokens / wall time (includes ramp). |
| c4_steady agg_tps | Tokens from all 4 streams inside the window where all 4 are decoding. Use this for "N requests at once". |

Rules:
- Warm up after every restart. `perf.py` and `c4_steady.py` send one short request
  first by themselves. When timing by hand:
  `curl -s URL/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"<id>","messages":[{"role":"user","content":"hi"}],"max_tokens":8}'`.
- With thinking models, llama.cpp's chat endpoint renders 2 fewer prompt tokens
  (no `<think>\n`) than Loom. That is negligible for throughput, and goldens are
  unaffected.
- Every prompt carries a nonce, so prefix caching can't inflate prefill.
- Loom packages need `execution.rows ≥ 4` for the 4-stream numbers, and llama-server
  needs `-np 4`.
- llama.cpp numbers use its server-side `timings`, which exclude HTTP and
  tokenization. That slightly favors llama.cpp; say so.
- Measure on an idle machine: stop the demo servers and other models first
  (`pgrep -af "loom-serve|llama-server"`; Windows: `ps -W | grep -E 'loom-serve|llama-server'`).
- Noise levels:
  - decode and c4 vary by about ±2–3 %;
  - short prefill varies a lot, so use the median;
  - long prefill (8k–32k) single runs vary ±5–10 %, so repeat any metric you base
    a decision on.
- Don't use kernel microbenchmarks as headline numbers. The 32 MB MALL inflates
  them.

## 3. Reporting

**Run the audit first** (from `bench/`):
```bash
$T $S/audit.py .     # -> headline table + checks; exit 1 on FAIL; writes audit.json
```
It checks:
- **Baseline.** `llamacpp_pick.json` exists, and the measured baseline is the
  fastest build (decode and prefill).
- **Plausibility.** Loom is not above ~0.97 of a ceiling, and llama.cpp is not
  implausibly far below one.
- **Tracking.** `targets.jsonl` has a start row and later measurements.
- **Gates.** `compare.py` and `consistency.py` log to `gates.jsonl` by themselves.
  The latest compare must be PASS, and run on the final commit. Consistency must
  be PASS, or a documented near-tie: write the token, the gap and the cause in
  `consistency_note.md`.

Fix every FAIL before reporting, and report every WARN. The headline table is the
source for the report's numbers.

- **Lead with the bottom line**: a table with Loom and llama.cpp ROCm side by side
  for each metric, followed by one line on the correctness gate (e.g. "28 prompts
  up to 32k tokens; all divergences near-ties (max gap 0.97); concurrent ==
  sequential").
- **State the setup**:
  - hardware;
  - SDK wheel (CI artifact id or commit);
  - llama.cpp build tag and backend;
  - GGUF file and quant;
  - the package preset (context, pool, prefill chunk, rows).
- **Be honest about losses and limits.** Report the metrics where Loom is slower
  and any targets you missed (e.g. "prefill target 3,700 tok/s not reached:
  GEMM at ~50 % of WMMA peak"), plus what is unsupported (sampling, tools,
  images).
- **No "before this session" columns** in executive tables. Give current Loom
  against the current baseline.
- **Format.** Write executive reports as a single-page HTML artifact (Artifact tool;
  load the artifact-design skill first). The sections:
  1. methodology;
  2. functional;
  3. pressure: concurrency, long context and memory limits;
  4. quality;
  5. performance;
  6. an appendix of what was skipped and why.

  Diagrams should be drawn from the user's point of view, with readable font
  sizes (≥ 14 px). Generate the page from the results JSON with a small script, so no number is typed by hand.
- Keep raw logs (`perf_log.jsonl`, `compare.json`, `golden.json`) next to the
  report, so every number can be traced.
