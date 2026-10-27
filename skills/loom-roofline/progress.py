#!/usr/bin/env python3
"""Rate of approach to the roofline, and what to do next.

usage: progress.py bench/roofline.json bench/targets.jsonl --add VALUE --change "what" [--reverted] [--metric decode_tps]
       progress.py bench/roofline.json bench/targets.jsonl [--metric decode_tps]
                   [--budget-h 1] [--reserve-min 10] [--goal-tps N]
                   [--window-min 45] [--horizon-h H] [--stall-min M]

--budget-h: the user's time budget for the optimization campaign (default 1 h),
counted from the first "start" line in the log. The horizon is what is left of it,
minus --reserve-min for the final gates and report. --horizon-h overrides that.
--goal-tps: a user-supplied target, which replaces target/stretch as the goal.
--add: append a correctly formatted line (ts = now, commit = git HEAD) and then
evaluate. The first --add for a metric should be --change start. Prefer this to
writing the log by hand.

targets.jsonl: one line per measured build (kept or not), appended by the loop:
  {"ts": "2026-10-26 22:09:18", "metric": "decode_tps", "value": 59.87,
   "commit": "abc123", "change": "adaptive matvec rows", "kept": true}
A first line with "change": "start" marks the starting point. Time spent on
failed attempts counts: it is in the wall clock between lines.

Everything is in time per token (ms), where gains add: ceiling time, target time,
and the gap between the best kept build and them. The rate is gap closed per hour
over the trailing window. Verdicts:
  target/stretch count as met within half the noise band (1 %).
  CONTINUE  the projected time to the goal (target, or stretch once met) is within
            --horizon-h at the current rate.
  REVIEW    slower than that, or no gain above noise for --stall-min: run the plateau
            review (SKILL.md section 5). Pivot to the best alternative path or stop.
  BUDGET    the time budget (minus the reserve) is used up: run the final gates and report.
  DONE      the goal is met (the user's --goal-tps, or else the stretch), or the gap to the ceiling is below measurement noise.
"""
import argparse, json, subprocess
from datetime import datetime

NOISE = 0.02  # ±2 % run-to-run on decode/c4


def ms(tps):
    return 1000.0 / tps


def parse_ts(v):
    try:
        return datetime.fromtimestamp(float(v))     # epoch seconds
    except ValueError:
        return datetime.fromisoformat(str(v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roofline")
    ap.add_argument("log")
    ap.add_argument("--metric")
    ap.add_argument("--window-min", type=float, default=45)
    ap.add_argument("--budget-h", type=float, help="default: roofline.json campaign.budget_h, else 1")
    ap.add_argument("--reserve-min", type=float, default=10)
    ap.add_argument("--goal-tps", type=float)
    ap.add_argument("--horizon-h", type=float)
    ap.add_argument("--add", type=float)
    ap.add_argument("--change", default="")
    ap.add_argument("--reverted", action="store_true")
    ap.add_argument("--stall-min", type=float, help="default: budget/2, clamped to 20-90 min")
    a = ap.parse_args()
    rj = json.load(open(a.roofline))
    roof = {m["metric"]: m for m in rj["metrics"]}
    camp = rj.get("campaign", {})
    if a.budget_h is None:
        a.budget_h = camp.get("budget_h", 1.0)
    if a.stall_min is None:
        a.stall_min = min(90, max(20, a.budget_h * 30))
    if a.add is not None:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                                text=True).stdout.strip() or "-"
        with open(a.log, "a") as f:
            f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"),
                                "metric": a.metric or "decode_tps", "value": a.add,
                                "commit": commit, "change": a.change,
                                "kept": not a.reverted}) + "\n")
    rows = [json.loads(l) for l in open(a.log) if l.strip()]
    metrics = [a.metric] if a.metric else sorted({r["metric"] for r in rows})
    for m in metrics:
        R = roof[m]
        ev = sorted((r for r in rows if r["metric"] == m), key=lambda r: r["ts"])
        t = lambda r: parse_ts(r["ts"])
        kept = [r for r in ev if r.get("kept", True)]
        best = max(kept, key=lambda r: r["value"])
        now = t(ev[-1])
        t_ceil, t_tgt, t_str, t_now = ms(R["ceiling"]), ms(R["target"]), ms(R["stretch"]), ms(best["value"])
        met = lambda t_g: t_now <= t_g * (1 + NOISE / 2)       # within noise counts as met
        goal, t_goal = ("stretch", t_str) if met(t_tgt) else ("target", t_tgt)
        user_goal = a.goal_tps or camp.get("goals", {}).get(m)
        if user_goal:
            goal, t_goal, t_str = "user goal", ms(user_goal), ms(user_goal)
        # best kept value as of the window start
        w0 = now.timestamp() - a.window_min * 60
        before = [r for r in kept if t(r).timestamp() <= w0] or kept[:1]
        t_then = ms(max(r["value"] for r in before))
        span_h = max((now.timestamp() - max(w0, t(ev[0]).timestamp())) / 3600, 1e-6)
        rate = (t_then - t_now) / span_h                       # ms/token closed per hour
        frac = (t_then - t_now) / max(t_then - t_ceil, 1e-9) / span_h
        remaining = t_now - t_goal
        eta = remaining / rate if rate > 0 else float("inf")
        # last time the best value improved by more than noise
        last_gain, run = t(ev[0]), kept[0]["value"]
        for r in kept[1:]:
            if r["value"] > run * (1 + NOISE / 2):
                last_gain, run = t(r), r["value"]
        stalled = (now - last_gain).total_seconds() / 60 > a.stall_min
        start = t(next((r for r in ev if r.get("change") == "start"), ev[0]))
        used_h = (now - start).total_seconds() / 3600
        left_h = a.budget_h - used_h - a.reserve_min / 60
        horizon = a.horizon_h if a.horizon_h is not None else left_h
        early = span_h < 2 / 60 or len(ev) < 2
        if goal in ("stretch", "user goal") and met(t_str) or (t_now - t_ceil) / t_now < NOISE:
            verdict = "DONE"
        elif a.horizon_h is None and left_h <= 0:
            verdict = f"BUDGET ({used_h * 60:.0f} of {a.budget_h * 60:.0f} min used): run the final gates, report"
        elif early:
            verdict = "CONTINUE (too early to measure the rate)"
        elif stalled:
            verdict = f"REVIEW (no gain above noise for {(now - last_gain).total_seconds() / 60:.0f} min)"
        elif rate > 0 and eta <= horizon:
            verdict = "CONTINUE"
        else:
            verdict = f"REVIEW (ETA to {goal} exceeds the {horizon * 60:.0f} min left at this rate)"
        print(f"{m}: best {best['value']:.2f} ({best.get('change','')}) | ceiling {R['ceiling']:.1f} "
              f"target {R['target']:.1f} stretch {R['stretch']:.1f}")
        print(f"  gap to ceiling {t_now - t_ceil:.3f} ms/tok ({(t_now - t_ceil) / t_now:.0%} of step time); "
              f"to {goal} {max(remaining, 0):.3f} ms")
        if early:
            print("  rate: not enough history yet")
        else:
            print(f"  last {span_h * 60:.0f} min: closed {t_then - t_now:.3f} ms = {rate:.2f} ms/h "
                  f"({frac:.0%} of remaining gap per hour); ETA to {goal}: "
                  + (f"{eta * 60:.0f} min" if eta != float("inf") else "never at this rate"))
        print(f"  budget: {used_h * 60:.0f} min used of {a.budget_h * 60:.0f}; horizon {horizon * 60:.0f} min")
        print(f"  -> {verdict}")


if __name__ == "__main__":
    main()
