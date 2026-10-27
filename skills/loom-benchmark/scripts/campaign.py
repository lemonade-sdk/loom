"""Campaign bookkeeping that the measuring scripts do for you, so it can't be skipped.

- log_metrics(): perf.py / c4_steady.py append every Loom measurement to targets.jsonl
  (next to roofline.json) and print the rate-of-approach verdict from progress.py.
- log_gate(): compare.py / consistency.py append their verdict to gates.jsonl.
Both are no-ops for llama.cpp runs. log_metrics is a no-op until roofline.json exists.
"""
import json
import os
import subprocess
import sys
import time

PROGRESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "loom-roofline", "progress.py")


SCRIPTS = os.path.dirname(os.path.abspath(__file__))
ROOFLINE = os.path.join(SCRIPTS, "..", "..", "loom-roofline", "roofline.py")


def require(kind, log_path="perf_log.jsonl"):
    """Exit with the exact next command when a required campaign step was skipped."""
    d = os.path.dirname(os.path.abspath(log_path))
    if kind == "llama" and not os.path.exists(os.path.join(d, "llamacpp_pick.json")):
        sys.exit(f"ERROR: no llamacpp_pick.json in {d}. The baseline must be the fastest llama.cpp ROCm build "
                 f"on this machine. Run, from {d}:\n  {SCRIPTS}/pick_llamacpp.sh MODEL.gguf\n"
                 "then start the top-ranked build with llamacpp_server.sh and re-run this. "
                 "(--no-campaign for an ad-hoc measurement.)")
    if kind == "loom" and not find_roofline(log_path):
        sys.exit(f"ERROR: no roofline.json in {d}. Measure the llama.cpp baseline first (perf.py --kind llama "
                 f"into this same log), then run:\n  {sys.executable} {ROOFLINE} MODEL.gguf --cfg ../cfg.json "
                 f"--baseline {os.path.basename(log_path)} --out {d}/roofline.json --budget-h 1\n"
                 "(--no-campaign for an ad-hoc or ablation measurement.)")


def run_audit():
    """Run the definition-of-done audit after a gate, so the agent sees what is missing."""
    r = subprocess.run([sys.executable, os.path.join(SCRIPTS, "audit.py"), "."], capture_output=True, text=True)
    lines = [l for l in r.stdout.splitlines() if l.startswith(("[FAIL]", "[WARN]", "AUDIT"))]
    print("\n[audit] " + "\n[audit] ".join(lines or ["(audit produced no output)"]), flush=True)


def find_roofline(log_path="perf_log.jsonl"):
    for d in (os.path.dirname(os.path.abspath(log_path)), os.getcwd()):
        p = os.path.join(d, "roofline.json")
        if os.path.exists(p):
            return p
    return None


def _commit():
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or "-"


def log_metrics(values, label, reverted=False, log_path="perf_log.jsonl"):
    """values: {metric: tok/s} using roofline.json metric names."""
    roof = find_roofline(log_path)
    if not roof:
        print("[campaign] no roofline.json next to the log: run loom-roofline first "
              "(measurements are not being tracked)", flush=True)
        return
    known = {m["metric"] for m in json.load(open(roof))["metrics"]}
    tj = os.path.join(os.path.dirname(roof), "targets.jsonl")
    seen = set()
    if os.path.exists(tj):
        seen = {json.loads(l)["metric"] for l in open(tj) if l.strip()}
    logged = []
    with open(tj, "a") as f:
        for m, v in values.items():
            if m not in known or v is None:
                continue
            f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "metric": m, "value": v,
                                "commit": _commit(), "change": "start" if m not in seen else label,
                                "label": label, "kept": not reverted}) + "\n")
            logged.append(m)
    for m in logged:
        r = subprocess.run([sys.executable, PROGRESS, roof, tj, "--metric", m], capture_output=True, text=True)
        out = (r.stdout or r.stderr).strip().splitlines()
        print("[campaign] " + "\n[campaign] ".join(out), flush=True)


def log_gate(gate, verdict, **detail):
    with open("gates.jsonl", "a") as f:
        f.write(json.dumps({"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "gate": gate,
                            "verdict": verdict, "commit": _commit(), **detail}) + "\n")
