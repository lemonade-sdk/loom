#!/usr/bin/env python3
"""Definition of done for an optimization campaign. Run it from bench/ before reporting.

usage: audit.py [BENCH_DIR=.]    -> prints checks + the headline table, writes audit.json,
                                    exit 1 if any check FAILs

Checks, all from the files the scripts write themselves:
  baseline   llamacpp_pick.json exists, and the roofline baseline is its fastest build
  plausible  Loom never above ~0.97 of a ceiling (else the roofline undercounts);
             llama.cpp not implausibly far below it (else the baseline is suspect)
  tracking   targets.jsonl has a start row and later measurements
  gates      the latest compare is PASS; consistency PASS, or FAIL with consistency_note.md
             (near-tie with its cause); gates ran on the final build when git is in use
Use the headline table it prints in the report, so no number is typed by hand.
"""
import json
import os
import subprocess
import sys

d = sys.argv[1] if len(sys.argv) > 1 else "."
P = lambda f: os.path.join(d, f)
here = os.path.dirname(os.path.abspath(__file__))
PROGRESS = os.path.join(here, "..", "..", "loom-roofline", "progress.py")
checks = []


def check(name, ok, msg, level="FAIL"):
    checks.append({"check": name, "result": "ok" if ok else level, "detail": msg})


def jl(f):
    return [json.loads(l) for l in open(P(f)) if l.strip()] if os.path.exists(P(f)) else []


roof = json.load(open(P("roofline.json"))) if os.path.exists(P("roofline.json")) else None
pick = json.load(open(P("llamacpp_pick.json"))) if os.path.exists(P("llamacpp_pick.json")) else None
targets, gates = jl("targets.jsonl"), jl("gates.jsonl")
metrics = {m["metric"]: m for m in roof["metrics"]} if roof else {}

# baseline
check("baseline: picker ran", pick is not None and pick.get("ranked"),
      "llamacpp_pick.json missing: run pick_llamacpp.sh MODEL.gguf from bench/")
if pick and pick.get("ranked") and roof:
    top = pick["ranked"][0]
    base = metrics.get("decode_tps", {}).get("baseline")
    pbase = metrics.get("prefill_1000", {}).get("baseline")
    # chat vs raw /completion and different prompts: allow 8 % on decode, 25 % on prefill
    ok = (base is not None and base >= top["decode_tps"] * 0.92
          and (pbase is None or pbase >= top["prefill_1k_tps"] * 0.75))
    check("baseline: fastest build used", ok,
          f"roofline baseline decode {base} / prefill-1k {pbase} vs fastest build "
          f"{top['decode_tps']} / {top['prefill_1k_tps']} ({top['dir']}): re-measure with that build")
check("roofline: ran with a baseline", roof is not None and metrics.get("decode_tps", {}).get("baseline"),
      "roofline.json missing or without --baseline: run loom-roofline")

# plausibility, both directions
best = {}
for r in targets:
    if r.get("kept", True):
        best[r["metric"]] = max(best.get(r["metric"], 0), r["value"])
for m, R in metrics.items():
    if m in best:
        e = best[m] / R["ceiling"]
        check(f"plausible: Loom {m}", e <= 0.97,
              f"Loom at {e:.2f} of the ceiling: the roofline undercounts (sliding window, KV sharing, "
              "hybrid layers, expert count?)")
    be = R.get("baseline_eff")
    if be is not None:
        floor = 0.10 if m.startswith("prefill") else 0.30
        check(f"plausible: llama.cpp {m}", be >= floor,
              f"llama.cpp at {be:.2f} of the ceiling (< {floor}): suspect the baseline build or its settings",
              level="WARN")

# tracking
dec = [r for r in targets if r["metric"] == "decode_tps"]
check("tracking: progress logged", len(dec) >= 2 and dec[0].get("change") == "start",
      f"{len(dec)} decode_tps rows in targets.jsonl; perf.py logs them once roofline.json exists")

# gates
last = {}
for g in gates:
    last[g["gate"]] = g
c = last.get("compare")
check("gates: compare PASS", c is not None and c["verdict"] == "PASS",
      f"latest compare: {c and {k: c[k] for k in ('verdict', 'max_gap', 'bad') if k in c}}")
s = last.get("consistency")
note = os.path.exists(P("consistency_note.md"))
if s is not None and s["verdict"] != "PASS" and note:
    check("gates: consistency", False, "FAIL, documented as a near-tie in consistency_note.md: "
          "state it in the report", level="WARN")
else:
    check("gates: consistency", s is not None and s["verdict"] == "PASS",
          f"latest consistency: {s and s['verdict']} (a near-tie with a known cause needs consistency_note.md)")
kept = [r for r in targets if r.get("kept", True) and r.get("commit", "-") != "-"]
if kept and c and c.get("commit", "-") != "-":
    check("gates: ran on the final build", c["commit"] == kept[-1]["commit"],
          f"compare at {c['commit']}, last kept measurement at {kept[-1]['commit']}")

# headline table
rows = []
for m, R in metrics.items():
    if m not in best:
        continue
    goal = roof.get("campaign", {}).get("goals", {}).get(m) or R["target"]
    rows.append({"metric": m, "llama_cpp": R.get("baseline"), "loom": best[m],
                 "margin": round(best[m] / R["baseline"] - 1, 3) if R.get("baseline") else None,
                 "pct_of_ceiling": round(best[m] / R["ceiling"], 3), "target": goal,
                 "met": best[m] >= goal * 0.99})
print(f"{'metric':16} {'llama.cpp':>9} {'Loom':>9} {'margin':>7} {'%ceil':>6} {'target':>8}  met")
for r in rows:
    print(f"{r['metric']:16} {r['llama_cpp'] or 0:9.1f} {r['loom']:9.1f} "
          f"{(r['margin'] or 0):+7.0%} {r['pct_of_ceiling']:6.0%} {r['target']:8.1f}  {'yes' if r['met'] else 'no'}")
if pick and pick.get("ranked"):
    print(f"baseline build: {pick['ranked'][0]['dir']}")
print()
for ch in checks:
    print(f"[{ch['result']:4}] {ch['check']}" + ("" if ch["result"] == "ok" else f": {ch['detail']}"))
if roof and targets:
    for m in sorted({r["metric"] for r in targets}):
        r = subprocess.run([sys.executable, PROGRESS, P("roofline.json"), P("targets.jsonl"), "--metric", m],
                           capture_output=True, text=True)
        v = [l for l in r.stdout.splitlines() if "->" in l]
        print(f"[ctrl] {m}: {v[-1].strip() if v else r.stderr.strip()[:120]}")
json.dump({"checks": checks, "headline": rows}, open(P("audit.json"), "w"), indent=1)
fails = [c for c in checks if c["result"] == "FAIL"]
print("\nAUDIT", "FAIL" if fails else "PASS", f"({len(fails)} failing)")
sys.exit(1 if fails else 0)
