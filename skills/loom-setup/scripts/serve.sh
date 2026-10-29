#!/bin/bash
# Start loom-serve in the background and wait until every model is registered.
# usage: VENV=~/loom/.venv serve.sh NAME PORT DEVICE MEMORY_BYTES MODEL_JSON [MODEL_JSON...] [-- extra loom-serve flags]
#   DEVICE: amdgpu | 'xdna://0?columns=8' (NPU; add -- --pool_backing=fixed)
#   MEMORY_BYTES: always set it (e.g. 16000000000); it caps weights + KV pool + workspace.
# Logs/PIDs: ~/.cache/loom-serve/NAME.{log,pid,port} (pick unique NAMEs). Stop: stop.sh NAME
# Readiness = /v1/status registration "ready"; compile happens at startup (cached after the first run),
# weights load lazily on the first request. Works on Linux and in Git Bash on Windows.
set -u
name=$1; port=$2; dev=$3; mem=$4; shift 4
models=(); extra=()
while [ $# -gt 0 ]; do [ "$1" = "--" ] && { shift; extra=("$@"); break; }; models+=("--model=$1"); shift; done
venv=${VENV:-$HOME/loom/.venv}
bin=$venv/bin; [ -d "$bin" ] || bin=$venv/Scripts   # Windows venvs use Scripts/
py=$bin/python
state=${LOOM_SERVE_STATE:-$HOME/.cache/loom-serve}; mkdir -p "$state"   # NAME.pid / NAME.log / NAME.port live here
# Listening sockets on $port: `ss` on Linux, `netstat -ano` (last column = Windows PID) elsewhere.
listeners() { if command -v ss >/dev/null; then ss -ltnp | grep ":$port "; else netstat -ano -p tcp | tr -d '\r' | grep LISTENING | grep ":$port "; fi; }
owns_port() { if command -v ss >/dev/null; then listeners | grep -q "pid=$1,"; else listeners | grep -q " $(cat /proc/$1/winpid 2>/dev/null || echo $1)\$"; fi; }
if listeners >/dev/null; then echo "port $port busy:"; listeners; exit 1; fi
nohup "$bin/loom-serve" --device="$dev" --listen=127.0.0.1:$port --memory_bytes=$mem --workers=8 "${models[@]}" "${extra[@]}" > $state/$name.log 2>&1 &
echo $! > $state/$name.pid; echo $port > $state/$name.port; start=$(date +%s)
while kill -0 $(cat $state/$name.pid) 2>/dev/null; do
  st=$(curl -s -m 2 http://127.0.0.1:$port/v1/status | "$py" -c 'import sys,json; d=json.load(sys.stdin)["data"]; print(",".join(x["loom"]["registration"] for x in d) if d else "none")' 2>/dev/null)
  case "$st" in
    *failed*) echo "FAILED $name"; curl -s http://127.0.0.1:$port/v1/status | "$py" -c 'import sys,json; [print(x["id"], json.dumps(x["loom"].get("error") or x["loom"].get("compilation",{}).get("diagnostics"))[:3000]) for x in json.load(sys.stdin)["data"]]'
      # A failed registration keeps the process (and the port); stop it so a rebuild can retry.
      kill $(cat $state/$name.pid) 2>/dev/null; rm -f $state/$name.pid; exit 2;;
  esac
  if [ -n "$st" ] && [ "$st" != none ] && ! echo "$st" | tr ',' '\n' | grep -qv '^ready$'; then
    pid=$(cat $state/$name.pid)
    owns_port $pid || { echo "port $port answered but is not owned by pid $pid (stale server?)"; listeners; exit 3; }
    # Registration is ready once the program compiled (cached kernels make this ~1 s); weights load on the first request.
    echo "READY $name on :$port pid $pid after $(( $(date +%s) - start ))s ($st)"; exit 0
  fi
  sleep 1
done
echo "DIED $name"; grep -v heartbeat $state/$name.log | tail -n 40; exit 1
