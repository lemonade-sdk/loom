#!/bin/bash
# Start the Loom playground UI.
# usage: LOOM_UI_UPSTREAMS="gpu=http://127.0.0.1:8100,npu=http://127.0.0.1:8101" start_ui.sh [PORT=8090] [PYTHON_VENV]
# The venv needs fastapi, uvicorn and httpx (pip install fastapi uvicorn httpx).
port=${1:-8090}; venv=${2:-${UI_VENV:-$HOME/loom/.tools}}   # the tools venv created by loom-setup/install_loom.sh
here=$(cd "$(dirname "$0")" && pwd)
bin=$venv/bin; [ -d "$bin" ] || bin=$venv/Scripts   # Windows venvs use Scripts/
listeners() { if command -v ss >/dev/null; then ss -ltnp | grep ":$port "; else netstat -ano -p tcp | tr -d '' | grep LISTENING | grep ":$port "; fi; }
if listeners >/dev/null; then echo "port $port busy"; listeners; exit 1; fi
LOOM_UI_UPSTREAMS=${LOOM_UI_UPSTREAMS:-gpu=http://127.0.0.1:8100,npu=http://127.0.0.1:8101} \
  nohup "$bin/uvicorn" --app-dir "$here/ui" server:app --host 127.0.0.1 --port $port > /tmp/loom-ui-$port.log 2>&1 &
echo $! > /tmp/loom-ui-$port.pid
for i in $(seq 20); do curl -s -m 1 localhost:$port/api/targets >/dev/null && { echo "UI: http://127.0.0.1:$port  (targets: $(curl -s localhost:$port/api/targets))"; exit 0; }; sleep 0.5; done
echo "UI failed to start:"; tail -20 /tmp/loom-ui-$port.log; exit 1
