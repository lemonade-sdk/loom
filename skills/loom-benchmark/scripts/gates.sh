#!/bin/bash
# All correctness gates for a served package, in one run, with one verdict file.
# usage: [LOOM_ROOT=~/loom] [QUICK=1] LLAMACPP_DIR=DIR gates.sh PKG LOOM_URL GGUF BENCH_DIR [LLAMA_PORT=8130]
#   QUICK=1 (before / during optimization): needle at 50 % of the context only, one probe. Run the
#   full gate (no QUICK) once on the final build: it proves the whole context.
#   golden.py x2 (two llama.cpp recordings) + compare.py, consistency.py, agent_golden.py,
#   sampling_check.py and needle.py (50 % and 100 % of context, 2 probes each; llama.cpp joins the
#   needle comparison only when its slot covers the size, since a 262k llama.cpp prefill takes many
#   minutes). llama.cpp runs with one slot sized for the long goldens; it is stopped at the end.
# Writes BENCH_DIR/gates.json {"verdict": "PASS"|"FAIL", "gates": {name: {"pass": bool, "log": file}}}
# and BENCH_DIR/gates_*.log. Exit 1 when any gate fails. Runs sequentially: on a shared GPU the
# gates would otherwise compete with each other.
set -u
pkg=$(cd "$1" && pwd); url=$2; gguf=$3; bench=$(mkdir -p "$4" && cd "$4" && pwd); lport=${5:-8130}
S=$(cd "$(dirname "$0")" && pwd); root=${LOOM_ROOT:-$HOME/loom}
bin=$root/.tools/bin; [ -d "$bin" ] || bin=$root/.tools/Scripts; T=$bin/python
export PYTHONDONTWRITEBYTECODE=1 LLAMACPP_STATE=${LLAMACPP_STATE:-$bench/llamacpp-state}
ctx=$("$T" -c "import json;print(next(iter(json.load(open('$pkg/model.json'))['presets'].values()))['execution']['context_capacity'])")
model=$("$T" -c "import json;print(json.load(open('$pkg/llm_config.json'))['model_id'])")
# Long goldens that fit the context; llama.cpp slot covers them (and needle sizes up to 64k).
long=$("$T" -c "print(','.join(str(n) for n in (1000, 2000, 16000, 32000) if n + 2000 <= $ctx))")
slot=$("$T" -c "print(min($ctx, 65536) + 1024)")
cd "$bench"
declare -A result
run() {  # run NAME CMD...
  local name=$1; shift
  echo "== $name"; "$@" > "gates_$name.log" 2>&1; local rc=$?
  tail -4 "gates_$name.log"; result[$name]=$rc
}
LLAMACPP_DIR=${LLAMACPP_DIR:?set LLAMACPP_DIR to the llama.cpp ROCm build} "$S/llamacpp_server.sh" "$gguf" "$lport" 1 "$slot" || exit 1
L=http://127.0.0.1:$lport
run golden1 "$T" "$S/golden.py" "$pkg" golden.json --llama "$L" --long "$long"
run golden2 "$T" "$S/golden.py" "$pkg" golden2.json --llama "$L" --long "$long"
run compare "$T" "$S/compare.py" "$pkg" golden.json --alt golden2.json --url "$url"
grep -q '"verdict": "PASS"' gates_compare.log || result[compare]=1
run consistency "$T" "$S/consistency.py" "$url" --model "$model"
grep -q "^PASS$" gates_consistency.log || result[consistency]=1
run agent "$T" "$S/agent_golden.py" "$pkg" "$url" --llama "$L" --out agent_golden.json
run sampling "$T" "$S/sampling_check.py" "$pkg" "$url" --llama "$L"
needle_args=(--out needle.json)
[ "${QUICK:-0}" = 1 ] && needle_args+=(--sizes $(( (ctx - 2048) / 2 )) --probes 1)
if [ "$ctx" -le 65536 ]; then
  run needle "$T" "$S/needle.py" "$pkg" "$url" --llama "$L" "${needle_args[@]}"
else
  run needle "$T" "$S/needle.py" "$pkg" "$url" "${needle_args[@]}"
fi
kill "$(cat "$LLAMACPP_STATE/llamacpp-$lport.pid")" 2>/dev/null
"$T" - "${!result[@]}" -- "${result[@]}" <<'EOF'
import json, sys
args = sys.argv[1:]; i = args.index("--"); names, codes = args[:i], args[i + 1:]
gates = {n: {"pass": c == "0", "log": f"gates_{n}.log"} for n, c in zip(names, codes)}
verdict = "PASS" if all(g["pass"] for g in gates.values()) else "FAIL"
json.dump({"verdict": verdict, "gates": gates}, open("gates.json", "w"), indent=1)
print("GATES", verdict, {n: ("PASS" if g["pass"] else "FAIL") for n, g in sorted(gates.items())})
sys.exit(0 if verdict == "PASS" else 1)
EOF
