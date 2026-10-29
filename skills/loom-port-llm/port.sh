#!/bin/bash
# One command from a Hugging Face GGUF repo to a served, tested Loom package.
# usage: [LOOM_ROOT=~/loom] port.sh ORG/REPO[:FILE.gguf] WORKDIR [PORT=8160] [MEMORY_GIB=56] [MODEL_ID]
#   1. download the GGUF (FILE, else the repo's Q4_K_M, else Q8_0) and the base model's tokenizer.json
#   2. inspect_gguf.py: config + chat block derived from the template and verified (stops on any FAIL)
#   3. build.py at the native context, fitted to MEMORY_GIB
#   4. serve.sh (state in WORKDIR/state), kernel unit tests, smoke test (chat, thinking off, a tool call)
# Prints PORT_RESULT {...} with the package, model id, context and memory_bytes; the server keeps
# running (stop: LOOM_SERVE_STATE=WORKDIR/state stop.sh port). Exit codes: 2 download, 3 chat
# template needs a hand-written block, 4 build/serve, 5 kernel tests, 6 smoke test.
# Then run loom-benchmark/scripts/gates.sh for the correctness gates.
set -u
spec=$1; work=$(mkdir -p "$2" && cd "$2" && pwd); port=${3:-8160}; mem_gib=${4:-56}; model_id=${5:-}
here=$(cd "$(dirname "$0")" && pwd); K=$here/starter; S=$here/../loom-setup/scripts
root=${LOOM_ROOT:-$HOME/loom}
bin=$root/.tools/bin; [ -d "$bin" ] || bin=$root/.tools/Scripts
T=$bin/python; venv=$root/.venv; hf=$venv/bin/hf; [ -x "$hf" ] || hf=$venv/Scripts/hf
export LOOM_SERVE_STATE=$work/state PYTHONDONTWRITEBYTECODE=1
repo=${spec%%:*}; file=""; [ "$spec" != "$repo" ] && file=${spec#*:}
cd "$work"
step() { echo "== $*"; }

step "1/6 download"
if [ -z "$file" ]; then
  file=$("$T" - "$repo" <<'EOF'
import json, sys, urllib.request
files = [f["path"] for f in json.load(urllib.request.urlopen(f"https://huggingface.co/api/models/{sys.argv[1]}/tree/main?recursive=true"))
         if f["path"].lower().endswith(".gguf") and "mmproj" not in f["path"].lower()]
for q in ("Q4_K_M", "Q4_K_S", "Q5_K_M", "Q6_K", "Q8_0"):
    hit = [f for f in files if q.lower() in f.lower() and "-of-" not in f]
    if hit:
        print(sorted(hit, key=len)[0]); break
EOF
)
fi
[ -n "$file" ] || { echo "no supported single-file GGUF (Q4_K_M/Q4_K_S/Q5_K_M/Q6_K/Q8_0) in $repo"; exit 2; }
"$hf" download "$repo" "$file" --local-dir gguf >/dev/null || { echo "download of $repo/$file failed"; exit 2; }
gguf=$work/gguf/$file
# Tokenizer: the base model named in the GGUF, else the repo name without -GGUF.
tokrepos=$("$T" - "$gguf" "$repo" <<'EOF'
import re, sys
from gguf import GGUFReader
r = GGUFReader(sys.argv[1]); out = []
for k, f in r.fields.items():
    if re.fullmatch(r"general\.base_model\.\d+\.repo_url", k):
        out.append(f.contents().split("huggingface.co/")[-1].strip("/"))
repo = sys.argv[2]; base = re.sub(r"[-_.]gguf$", "", repo, flags=re.I)
out += [base, repo]
org, name = base.split("/", 1)
out += [f"Qwen/{name}", f"{org}/{name}-Instruct"]
print(" ".join(dict.fromkeys(out)))
EOF
)
mkdir -p tok
for tr in $tokrepos; do
  "$hf" download "$tr" tokenizer.json --local-dir tok >/dev/null 2>&1 && { echo "tokenizer from $tr"; break; }
done
[ -f tok/tokenizer.json ] || { echo "no tokenizer.json found in: $tokrepos (download it from the base model into $work/tok)"; exit 2; }
[ -n "$model_id" ] || model_id=$(echo "$tokrepos" | cut -d' ' -f1)

step "2/6 inspect (chat block derived from the template and verified)"
"$T" -I "$K/inspect_gguf.py" "$gguf" cfg.json | tee inspect.log | grep -v "^chat OK"
if grep -q "^chat FAIL\|not derivable\|WARNING: no chat block\|WARNING: no kernels" inspect.log; then
  echo "STOP: see inspect.log; fix cfg.json (chat block / formats) by hand as loom-port-llm/SKILL.md describes, then rerun build.py"; exit 3
fi

step "3/6 build"
"$T" "$K/build.py" cfg.json pkg --tokenizer tok/tokenizer.json --model-id "$model_id" --memory-gib "$mem_gib" | tee build.log || exit 4
mem=$(grep -o -- "--memory_bytes=[0-9]*" build.log | cut -d= -f2)
ctx=$(grep -o "^context [0-9]*" build.log | cut -d' ' -f2)

step "4/6 serve"
"$S/stop.sh" port >/dev/null 2>&1
VENV=$venv "$S/serve.sh" port "$port" amdgpu "$mem" pkg/model.json || exit 4

step "5/6 kernel unit tests"
"$T" -I "$K/make_kernel_tests.py" cfg.json ktest >/dev/null && VENV=$venv "$K/run_kernel_tests.sh" pkg ktest | tee ktest.log | tail -3
grep -q "^FAIL\|Traceback" ktest.log && { echo "STOP: kernel tests failed (ktest.log); see the bring-up debugging order in SKILL.md"; exit 5; }

step "6/6 smoke test"
"$T" - "$work/pkg" "http://127.0.0.1:$port" <<'EOF' || exit 6
import json, sys, requests
pkg, url = sys.argv[1], sys.argv[2]
cfg = json.load(open(pkg + "/llm_config.json")); model = cfg["model_id"]; chat = cfg["chat"]
off = {"chat_template_kwargs": {"enable_thinking": False}} if chat.get("thinking") == "toggle" else {}
post = lambda body: requests.post(url + "/v1/chat/completions", json={"model": model, **body}, timeout=600).json()
ok = True
r = post({"messages": [{"role": "user", "content": "What is the capital of France? Answer in one word."}], "max_tokens": 600, "temperature": 0})
m = r["choices"][0]["message"]; good = "paris" in ((m.get("content") or "") + (m.get("reasoning_content") or "")).lower()
print("chat:", "OK" if good else "FAIL", repr((m.get("content") or "")[:80])); ok &= good
tools = [{"type": "function", "function": {"name": "get_weather", "description": "Current weather for a city.",
          "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
r = post({"messages": [{"role": "user", "content": "What's the weather in Paris right now? Use the tool."}], "tools": tools,
          "max_tokens": 600, "temperature": 0, **off})
calls = r["choices"][0]["message"].get("tool_calls") or []
good = bool(calls) and calls[0]["function"]["name"] == "get_weather" and "paris" in calls[0]["function"]["arguments"].lower()
print("tool call:", "OK" if good else "FAIL", json.dumps(calls)[:200]); ok &= good
n = requests.post(url + "/v1/chat/completions/input_tokens", json={"model": model, "messages": [{"role": "user", "content": "hi"}]}).json()
print("input_tokens:", n)
sys.exit(0 if ok else 1)
EOF
echo "PORT_RESULT $(printf '{"package": "%s", "model_id": "%s", "context": %s, "memory_bytes": %s, "port": %s, "gguf": "%s"}' "$work/pkg" "$model_id" "${ctx:-0}" "$mem" "$port" "$gguf")"
