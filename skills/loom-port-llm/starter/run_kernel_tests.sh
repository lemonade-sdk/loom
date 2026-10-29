#!/bin/bash
# usage: VENV=~/loom/.venv run_kernel_tests.sh PACKAGE TESTDIR   (TESTDIR from make_kernel_tests.py)
pkg=$(cd "$1" && pwd); dir=$2; venv=${VENV:-$HOME/loom/.venv}
bin=$venv/bin; [ -d "$bin" ] || bin=$venv/Scripts   # Windows venvs use Scripts/
cd "$dir" && cat "$pkg/program/llm.loom" cases.loom-test > combined.loom
"$bin/loom-test" combined.loom --device=${DEVICE:-amdgpu} > out.json 2> err.txt
grep -v '^{"format' err.txt | head -20
"$bin/python" - <<'PY'
import json
d = json.load(open("out.json")); bad = 0
for s in d["samples"]:
    fails = [f.get("detail", f.get("message", "")) if isinstance(f, dict) else str(f) for f in s["expectations"].get("failures", [])]
    issues = [i.get("message", "") for i in s.get("issues", [])]
    bad += not s["passed"]
    print(("PASS " if s["passed"] else "FAIL ") + s["case"], *(x[:300] for x in issues + fails[:2]))
print(f"{len(d['samples']) - bad}/{len(d['samples'])} passed")
PY
