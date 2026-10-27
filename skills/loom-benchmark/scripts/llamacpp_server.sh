#!/bin/bash
# Start the llama.cpp ROCm reference server (lemonade-sdk/llamacpp-rocm build) on a GGUF.
# usage: LLAMACPP_DIR=~/llamacpp-rocm llamacpp_server.sh MODEL.gguf [PORT=8130] [PARALLEL=4] [CTX_PER_SLOT=8192]
#   golden / long-context runs: PARALLEL=1 CTX_PER_SLOT=34000
#   stop: kill $(cat /tmp/llamacpp-$PORT.pid)
set -e
gguf=$1; port=${2:-8130}; np=${3:-4}; per=${4:-8192}
dir=${LLAMACPP_DIR:?set LLAMACPP_DIR to the unzipped llamacpp-rocm release}
if curl -s -m 2 localhost:$port/health >/dev/null 2>&1 || ss -ltn | grep -q ":$port "; then
  echo "port $port is busy (another llama-server or loom-serve?)"; ss -ltnp | grep ":$port " || true; exit 1
fi
cd "$dir"
LD_LIBRARY_PATH=$PWD nohup ./llama-server -m "$gguf" -ngl 99 -c $((np * per)) -np $np --port $port --host 127.0.0.1 > /tmp/llamacpp-$port.log 2>&1 &
echo $! > /tmp/llamacpp-$port.pid
for i in $(seq 300); do
  curl -s localhost:$port/health | grep -q ok && { echo "READY llama.cpp on :$port (pid $(cat /tmp/llamacpp-$port.pid))"; exit 0; }
  kill -0 $(cat /tmp/llamacpp-$port.pid) 2>/dev/null || { echo "llama-server died:"; tail -20 /tmp/llamacpp-$port.log; exit 1; }
  sleep 1
done
echo "timeout"; exit 1
