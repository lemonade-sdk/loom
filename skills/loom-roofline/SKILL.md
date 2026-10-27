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
| eff | llama.cpp's fraction of the ceiling. **This, not Loom, decides how big the win can be.** At eff 0.75, the most Loom can gain is about 1.3×. At eff 0.15, 3–6× is on the table, and such results are routinely reported for new architectures, MoE and prefill. |
| TARGET | ceiling × the **best efficiency a Loom port has actually reached** on this hardware (`EFF` in the script; sources in `reference/calibration.md`). Realistic by construction. |
| margin | TARGET ÷ llama.cpp − 1. Reported as is, with no cap and no label. |
| stretch | ceiling × `STRETCH`: what the known but not yet built levers should give (GEMM efficiency already seen in isolation, small-op fusion, batched expert gather). |
| at-ceiling | Under 3 % headroom: llama.cpp is near the roofline. Target parity and spend the effort elsewhere. |

**Sanity-check the model before trusting it.** If llama.cpp's `eff` is above about
0.95 on any metric, the roofline undercounts. Check for a missed sliding window, KV
sharing, a gathered table counted in full, or a wrong expert count. Fix that before
setting targets.

## 3. Pick the targets: realistic, achievable, impressive

- **Realistic.** TARGET comes from an efficiency some port has already reached
  here, so it is not a guess. Don't shade it down because the margin "looks too
  big". A large margin means llama.cpp is leaving the hardware idle.
- **Impressive.** Report each metric as a margin over llama.cpp ROCm. Choose a
  **headline set** of 2–4 metrics that matter for the use case:
  - decode tok/s always;
  - prefill at 1k–2k;
  - c4 if serving many users;
  - long-context decode/prefill if the model is advertised for long context.

  Lead with the metrics that have the most headroom. Drop at-ceiling metrics from
  the headline and set them to "≥ parity".
- **Achievable.** Use the gap budget below to name the lever for each headline
  metric, and mark it **tuning** (existing kernels, retuned or fused) or **new
  kernel** (something the kit lacks: flash attention for head_dim ≠ 128, a new
  quant format, an MoE gather, GEMM pipelining). A new-kernel target is achievable
  when that kernel is built. Size it, schedule it, and don't lower the target to
  avoid it.
- **Stretch.** Once a metric meets TARGET, keep going toward stretch while the stop
  rules allow. Any result above TARGET raises `EFF` for every later port.
- Don't ask the user to choose. Announce the targets in one short table (metric,
  llama.cpp, target, margin, stretch, lever, tuning/new kernel), then start working.
  The user can override them.
- If the user *did* give a target, still run this. If their target is above the
  stretch ceiling, say so once with the numbers. If it is below TARGET, say how much
  more is available and aim for TARGET.

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
- **Budget per lever, not per margin.** As a rule of thumb, a tuning lever gets
  about 4 attempts. A new kernel gets one build, its A/B test, and about 4 tuning
  attempts. A metric with a 3× margin and one missing kernel is a normal plan, not
  an overrun. To go past a lever's budget, you need a measured reason, such as an
  ablation showing that the lever still covers ≥ the remaining gap.
- **Stretch.** Pursue stretch only on metrics that met TARGET, and only while each
  attempt still gains ≥ 1 %.
- **Never** stop with a correctness gate failing. The last kept build must pass
  `compare.py` + `consistency.py`.

## 6. Report and recalibrate

- Report each headline metric as: target, stretch, achieved, % of ceiling, margin
  vs llama.cpp, and met / short (with the reason).
- If a port reached a higher efficiency than `EFF`, update `EFF` in `roofline.py`
  and add the row to `reference/calibration.md`. Do the same for `STRETCH` if a
  port beat it. This is how the targets get more ambitious over time.
- If a target was missed, also add the row. A metric type (decode, prefill, c4) that keeps landing short means
  its efficiency is too optimistic.
