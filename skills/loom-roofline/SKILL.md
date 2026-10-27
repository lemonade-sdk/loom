---
name: loom-roofline
description: Roofline analysis for a Loom LLM port on Strix Halo — compute the hardware ceiling for decode, long-context decode, prefill and 4-stream throughput from the GGUF, compare llama.cpp ROCm's efficiency, and pick realistic, achievable, impressive per-metric optimization targets with a stopping rule. Use before any open-ended "optimize this model" / "make it fast" / "beat llama.cpp" request, when the user gives no target, or when asked "what's possible" / "is this target achievable".
---

# Roofline → targets

The point is to decide **once, up front, with numbers**, what the port should reach,
so the optimization loop (`loom-optimize`) has a finish line, and so a stopping
decision is a measurement rather than a debate.

## 1. Inputs (all from the same session, on an idle machine)

- The GGUF and the port's `cfg.json`. The cfg supplies sliding layers, KV sharing
  (`kv_layers`) and `head_dim_swa`, which the GGUF may not spell out.
- llama.cpp ROCm numbers on the same GGUF, from `perf.py` / `c4_steady.py`
  (`loom-benchmark`). Write them as `baseline.json`:
  `{"decode_tps":…, "decode_tps_1k":…, "decode_tps_2k":…, "prefill_1000":…, "prefill_2000":…, "c4_steady":…}`.
  For long context add `decode_tps_16k`, `prefill_16000`, and so on.
- The current Loom numbers, if the port already exists.

## 2. Run

```bash
T=~/loom/.tools/bin/python; R=${CLAUDE_SKILL_DIR}
$T $R/roofline.py MODEL.gguf --cfg cfg.json --baseline baseline.json --ctx 1000,2000,16000 --out bench/roofline.json
```
The script prints the bytes read per decode token, broken down by kind, so you can
see where decode time must go:
- head;
- attention;
- FFN;
- active experts;
- gathered embeddings, counted as 1 row per token, *not* the whole table.

Per metric it then prints a row:

| Column | Meaning |
|---|---|
| ceiling | Hardware roofline. Decode: 216 GB/s ÷ (weights + KV read per token). Prefill: 59 TFLOP/s ÷ (2·params + causal attention flops) per token. c4: 4 tokens per weight pass. For MoE, the 4 tokens route to their own experts, so those weights are read for each token. |
| practical | ceiling × the **best efficiency a Loom port has actually reached** on this hardware (`EFF` in the script; sources in `reference/calibration.md`). |
| eff | llama.cpp's fraction of the ceiling. |
| headroom | practical ÷ llama.cpp − 1. |
| class | large ≥ 25 %, moderate 10–25 %, tight 3–10 %, at-ceiling < 3 %. |
| TARGET | The practical ceiling, or parity when the class is at-ceiling. |

**Sanity-check the model before trusting it.** If llama.cpp's `eff` is above about
0.95 on any metric, the roofline undercounts. Check for a missed sliding window, KV
sharing, a gathered table counted in full, or a wrong expert count. Fix that before
setting targets.

## 3. Pick the targets: realistic, achievable, impressive

- **Realistic.** A target never exceeds the practical ceiling. Some port has
  already reached that efficiency here, so it is not a guess.
- **Impressive.** State each target as a margin over llama.cpp ROCm. Choose a
  **headline set** of 2–4 metrics that matter for the use case:
  - decode tok/s always;
  - prefill at 1k–2k;
  - c4 if serving many users;
  - long-context decode/prefill if the model is advertised for long context.

  Drop at-ceiling metrics from the headline and set them to "≥ parity".
- **Achievable.** For each headline metric, write down the lever, using the gap
  budget below. A large-class target that needs a kernel the kit doesn't have
  (flash attention for head_dim ≠ 128, a new quant format, MoE gather) is achievable
  only with that kernel. Say so, and size the work.
- Don't ask the user to choose. Announce the targets in one short table (metric,
  llama.cpp, target, margin, class, lever), then start working. The user can
  override them.
- If the user *did* give a target, still run this. If their target is above the
  practical ceiling, say so once with the numbers, and work toward the practical
  ceiling.

## 4. Gap budget: convert the target into milliseconds

For a throughput metric, `gap_ms = 1000/current − 1000/target` per token. For
decode, the time at the roofline is `1000/ceiling`; everything above it is overhead.
Attribute that overhead by ablation (`loom-optimize` step 2: double or skip a
kernel, rebuild, measure). Typical buckets:
- matvecs under-filling the GPU, which is fixed with rows-per-workgroup / wave shape;
- attention;
- small-op kernels (norms, rope, PLE, gates);
- the head;
- per-step fixed cost.

**Feasibility rule.** If the overhead you can plausibly remove, summed over the
ablated kernels with the levers you have, is smaller than `gap_ms`, the target is
not achievable this session. Lower it to what the identified levers give. Record
the new target and the reason in the log.

## 5. Stop rules (no "working all night")

Track each attempt in `bench/targets.jsonl` as one JSON line:
`{metric, target, before, after, change, kept}`.
- **Met.** Stop work on a metric as soon as two consecutive runs meet its target,
  and the correctness gates pass.
- **Stalled.** After 3 attempts in a row that each improve the target metric by
  less than 1 %, re-ablate. After 2 re-ablations with no new lever, stop that metric
  and keep the best build.
- **Budget.** As a rule of thumb:
  - moderate: about 6 attempts;
  - large: about 10 attempts, or one new kernel;
  - tight: about 4 attempts.

  Going past the budget needs a measured reason, such as a lever whose ablation
  shows ≥ the remaining gap.
- **Never** stop with a correctness gate failing. The last kept build must pass
  `compare.py` + `consistency.py`.

## 6. Report and recalibrate

- Report each headline metric as: target, achieved, % of ceiling, margin vs
  llama.cpp, and met / short (with the reason).
- If a port reached a higher efficiency than `EFF`, update `EFF` in `roofline.py`
  and add the row to `reference/calibration.md`. This is how the targets get more
  ambitious over time.
- If a target was missed, also add the row. A class that keeps landing short means
  its efficiency is too optimistic.
