---
name: loom-roofline
description: Roofline analysis for a Loom LLM port on Strix Halo — compute the hardware ceiling for decode, long-context decode, prefill and 4-stream throughput from the GGUF, compare llama.cpp ROCm's efficiency, and pick realistic, achievable, impressive per-metric optimization targets with a stopping rule. Use before any open-ended "optimize this model" / "make it fast" / "beat llama.cpp" request, when the user gives no target, or when asked "what's possible" / "is this target achievable".
---

# Roofline → targets

The point is to decide **once, up front, with numbers**, what the port should reach,
so the optimization loop (`loom-optimize`) has a finish line, and so a stopping
decision is a measurement rather than a debate.

## 0. Effort: what the user asked for

Read the request for two optional inputs: a **performance target** and a **time
budget**. The time budget is the user's wall-clock time for the optimization
campaign, from the roofline run to the final report, including builds, gates and
reverted attempts. Porting and baseline measurement are prerequisites and are not
counted, but report how long they took.

| User said | Goal | Budget (`--budget-h`) |
|---|---|---|
| Nothing ("optimize this model") | The roofline TARGET per headline metric, then stretch | **1 h**: return a reasonable degree of optimization in an hour |
| A time only ("spend 3 hours", "quick pass", "overnight") | TARGET, then stretch | As stated. "Quick" ≈ 0.5 h, "thorough" ≈ 4 h, "overnight" ≈ 8 h, "don't work all night" caps it |
| A target only ("+50 % decode", "70 tok/s") | The user's target (`--goal-tps`) | No fixed cap. The rate controller still stops when no path can reach the target. If the projected time is more than ~2 h, say so with the numbers as soon as it is known |
| Both | The user's target | As stated. If the projected time exceeds the budget, say so early: give what the budget buys, and what the target would take |

In every case:
- Check a user target against the roofline once. If it is above the stretch, say
  so, with the ceiling.
- If a user target is below TARGET, say how much more is available, then honor
  their target as the finish line.
- Within a budget, spend time where the **expected gain per hour** is highest
  across the headline metrics. New kernels count too. On Bolt-E4B, a flash prefill
  kernel for head_dim 256/512 took about 30 min and gave +19–28 % prefill inside a
  1 h budget.
- **Catalog levers first.** Levers in `loom-optimize/reference/optimization-catalog.md`
  that have a measured gain on a similar model are the cheapest ms/h there is. Apply
  them in catalog order before inventing new ones. In run 2, skipping fused QKV and
  paired-row gate/up, and trying two untested ideas instead, cost about 1–2 % decode.
- Keep a reserve (default 10 min) at the end for the final `compare.py` +
  `consistency.py`, the server restart and the report. Never overrun the budget by
  starting an attempt that cannot finish its gates in time.

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
$T $R/roofline.py MODEL.gguf --cfg cfg.json --baseline baseline.json --ctx 1000,2000,16000 --out bench/roofline.json \
   --budget-h 1            # the user's budget (section 0); add --goal decode_tps=70 for a user target
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

## 5. Steer by rate of approach (diminishing returns)

Optimization returns diminish, so the question is never "is there more to gain?"
(there always is, up to the ceiling). The question is **how fast this path is
closing the gap**, compared with the other paths available.

**Tracking is automatic.** Once `bench/roofline.json` exists, every Loom run of
`perf.py` (run from `bench/`) and `c4_steady.py --label NAME` does three things:
1. appends its numbers to `bench/targets.jsonl`, with the time and the git commit;
2. marks the first measurement of each metric as the start;
3. prints the controller verdict for each metric (`[campaign] … -> CONTINUE`).

Label every build (`perf.py URL LABEL`), and pass `--reverted` for a build you
discard: its time still counts. The budget and any user goals come from
`roofline.json` (set with `--budget-h` / `--goal`). To run the controller by hand:
```bash
$T $R/progress.py bench/roofline.json bench/targets.jsonl --metric decode_tps
```
It works in **ms per token** (gains add; percentages don't), as follows:
1. It takes the gap from the best kept build to the ceiling, the target and the
   stretch.
2. It measures the wall-clock rate at which the gap has closed over the trailing
   window. Time spent on reverted attempts counts.
3. It projects the time to the current goal: the target, or the stretch once the
   target is met within noise.

Act on the verdict:
The horizon is what is left of the budget, minus the reserve.
- **CONTINUE**: at the current rate the goal is reached within the horizon. Keep
  going on this path.
- **REVIEW**: the ETA exceeds the horizon, or there has been no gain above noise
  for `--stall-min`. Do the plateau review below before any further attempt.
- **BUDGET**: the time is used up. Run the final gates on the best build, serve it
  and report.
- **DONE**: the goal is met (the user's target, or else the stretch), or the gap to the ceiling is within noise. Move on
  to the next headline metric.

On this session's Bolt decode run (ceiling 14.68 ms/token), with the default 1 h
budget, the replay goes:
- at 16 minutes, the first lever had closed 1.74 ms, 46 % of the gap (6.6 ms/h):
  CONTINUE;
- at 36 minutes the target was met, but the stretch was 21 minutes away at the
  current rate with only 14 left: REVIEW. Pivot to a cheap lever, or wrap up;
- past the budget: BUDGET.

With a 4 h budget the same log says CONTINUE. This replaces the session's
back-and-forth about feasibility with a measurement.

### Plateau review (REVIEW verdict): pivot or stop

1. **Re-ablate the current best build.** Diminishing returns usually mean the cost
   has moved. The kernel you were optimizing may no longer be the biggest one.
2. **List candidate paths, each with an expected rate:**
   `expected ms/h = (ms the lever can remove, from the ablation × the fraction
   plausibly removable) ÷ (estimated hours, including build, A/B and gates)`.
   Draw the candidates from:
   - the rest of the ablation table;
   - the `loom-optimize` catalog;
   - a **new kernel** (section 3);
   - **forks from prior decisions**: reopen an option from `bench/decisions.jsonl`
     whose rejection reason no longer holds. For example, fusing the norm into the
     matvec lost when the matvec was under-filled, and may win now that it isn't.
     To fork, branch from that decision's commit (`git worktree add ../fork-X <commit>`),
     then build and serve on a separate port with `tools/variant.sh`. Choose an
     earlier commit when a later choice blocks the lever, for example a layout that
     prevents a fusion.
3. **Pick the best candidate** if its expected rate gives an ETA to the goal within
   the horizon. Log the choice. A switch of path resets the stall timer, but not
   the measured rate.
4. **Stop the metric** if no candidate meets the horizon. Keep the best build and
   record in the report why the remaining gap isn't worth it, with the numbers:
   the ablation table and the best candidate's expected rate.

**Record decision points** in `bench/decisions.jsonl`, written as you go and not
reconstructed at the end, whenever you choose one
design over another:
`{"ts", "commit", "decision", "chosen", "rejected": [{"option", "measured", "why"}]}`.
The rejected options, with their measured numbers and the reason, are what makes a
fork cheap to evaluate later instead of re-deriving it.

Rules that override the rate:
- **Never** stop with a correctness gate failing. The last kept build must pass
  `compare.py` + `consistency.py`. The time spent on gates counts toward the rate.
- Re-measure a value near the target before calling it met or missed (noise ±2 %).
- The stall timer defaults to half the budget, clamped to 20–90 min. A long budget
  tolerates a longer exploration, such as building a new kernel, before a review.

## 6. Report and recalibrate

Run `audit.py` from `bench/` before writing the report. It is the definition of
done; see `loom-benchmark`. It must print `AUDIT PASS`. Use its headline table
(llama.cpp, Loom, margin, % of ceiling, target, met) in the report.

- Report each headline metric as: target, stretch, achieved, % of ceiling, margin
  vs llama.cpp, and met / short (with the reason). Include the approach curve (gap
  vs elapsed time, from `targets.jsonl`) and the stop or pivot decisions.
- If a port reached a higher efficiency than `EFF`, update `EFF` in `roofline.py`
  and add the row to `reference/calibration.md`. Do the same for `STRETCH` if a
  port beat it. This is how the targets get more ambitious over time.
- If a target was missed, also add the row. A metric type (decode, prefill, c4) that keeps landing short means
  its efficiency is too optimistic.
