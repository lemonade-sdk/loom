#!/bin/bash
# Find every llama.cpp ROCm (HIP) build on this machine, benchmark each on MODEL.gguf,
# and print them fastest first. The baseline is the strongest build available:
# version strings are unreliable (the lemonade nightly prints "version: 1") and builds
# differ a lot (Bolt-E4B c4: b1274 125 vs b1355 157; LFM2.5-24B prefill: 1062 vs 2140).
# usage: pick_llamacpp.sh MODEL.gguf [PORT=8139] [SEARCH_ROOT=$HOME]
#   writes the ranking to ./llamacpp_pick.json (run it from bench/; audit.py checks it)
set -u
gguf=$1; port=${2:-8139}; root=${3:-$HOME}
here=$(cd "$(dirname "$0")" && pwd)
py=${PY:-$HOME/loom/.tools/bin/python}
doc=$here/../data/doc_1000.txt
mapfile -t dirs < <(find "$root" -maxdepth 7 -name llama-server -type f 2>/dev/null \
  | xargs -r -n1 dirname | while read -r d; do [ -e "$d/libggml-hip.so" ] && echo "$d"; done \
  | while read -r d; do echo "$(readlink -f "$d/llama-server" | xargs sha256sum | cut -c1-16) $d"; done \
  | sort -u -k1,1 | cut -d' ' -f2-)
[ ${#dirs[@]} -gt 0 ] || { echo "no llama.cpp ROCm build found (no libggml-hip.so next to llama-server)"; exit 1; }
echo "candidates (deduplicated by binary hash): ${#dirs[@]}"
res=()
for d in "${dirs[@]}"; do
  LLAMACPP_DIR=$d bash "$here/llamacpp_server.sh" "$gguf" "$port" 1 8192 >/dev/null 2>&1 || { echo "  $d: failed to start"; continue; }
  line=$("$py" - "$port" "$doc" <<'E'
import json, sys, requests
port, doc = sys.argv[1], open(sys.argv[2]).read()
u = f"http://127.0.0.1:{port}/completion"
requests.post(u, json={"prompt": "hi", "n_predict": 8})                  # warm-up
d = requests.post(u, json={"prompt": "Write an essay about rivers.", "n_predict": 200,
                           "temperature": 0, "cache_prompt": False}).json()["timings"]
p = requests.post(u, json={"prompt": "nonce 7\n" + doc + "\nSummarize.", "n_predict": 4,
                           "cache_prompt": False}).json()["timings"]
print(f"{d['predicted_per_second']:.1f} {p['prompt_per_second']:.0f}")
E
)
  kill "$(cat /tmp/llamacpp-$port.pid)" 2>/dev/null; sleep 2
  echo "  decode ${line% *} tok/s  prefill-1k ${line#* } tok/s  $d"
  res+=("$line $d")
done
echo "ranked by decode, then prefill (use the top one; report its path/tag):"
printf '%s\n' "${res[@]}" | sort -k1,1nr -k2,2nr | awk '{printf "  %7s %7s  %s\n",$1,$2,$3}'
printf '%s\n' "${res[@]}" | sort -k1,1nr -k2,2nr | "$py" -c '
import json, sys
rows = [dict(zip(("decode_tps", "prefill_1k_tps", "dir"), l.split(None, 2))) for l in sys.stdin if l.strip()]
for r in rows: r["decode_tps"], r["prefill_1k_tps"] = float(r["decode_tps"]), float(r["prefill_1k_tps"])
json.dump({"gguf": sys.argv[1], "ranked": rows}, open("llamacpp_pick.json", "w"), indent=1)
print("wrote llamacpp_pick.json; baseline:", rows[0]["dir"] if rows else None)' "$gguf"
