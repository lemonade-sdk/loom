#!/bin/bash
# Start the llama.cpp ROCm reference server (lemonade-sdk/llamacpp-rocm build) on a GGUF.
# usage: LLAMACPP_DIR=~/llamacpp-rocm llamacpp_server.sh MODEL.gguf [PORT=8130] [PARALLEL=4] [CTX_PER_SLOT=8192]
#   golden / long-context runs: PARALLEL=1 CTX_PER_SLOT=34000
#   stop: kill $(cat /tmp/llamacpp-$PORT.pid)
# Refuses a build that does not see a ROCm device: on Windows, builds missing their HIP/BLAS DLLs
# silently fall back to CPU.
set -e
gguf=$1; port=${2:-8130}; np=${3:-4}; per=${4:-8192}
dir=${LLAMACPP_DIR:?set LLAMACPP_DIR to the unzipped llamacpp-rocm release}
listeners() { if command -v ss >/dev/null; then ss -ltnp | grep ":$port "; else netstat -ano -p tcp | tr -d '\r' | grep LISTENING | grep ":$port "; fi; }
if curl -s -m 2 localhost:$port/health >/dev/null 2>&1 || listeners >/dev/null; then
  echo "port $port is busy (another llama-server or loom-serve?)"; listeners || true; exit 1
fi
cd "$dir"
exe=./llama-server; [ -f $exe.exe ] && { exe=$exe.exe; command -v cygpath >/dev/null && gguf=$(cygpath -w "$gguf"); }
LD_LIBRARY_PATH=$PWD $exe --list-devices 2>&1 | grep -q ROCm || { echo "$dir/llama-server sees no ROCm device:"; LD_LIBRARY_PATH=$PWD $exe --list-devices 2>&1 | tail -5; exit 1; }
LD_LIBRARY_PATH=$PWD nohup $exe -m "$gguf" -ngl 99 -c $((np * per)) -np $np --port $port --host 127.0.0.1 > /tmp/llamacpp-$port.log 2>&1 &
echo $! > /tmp/llamacpp-$port.pid
for i in $(seq 300); do
  curl -s localhost:$port/health | grep -q ok && { echo "READY llama.cpp $($exe --version 2>&1 | grep -m1 version) on :$port (pid $(cat /tmp/llamacpp-$port.pid))"; exit 0; }
  kill -0 $(cat /tmp/llamacpp-$port.pid) 2>/dev/null || { echo "llama-server died:"; tail -20 /tmp/llamacpp-$port.log; exit 1; }
  sleep 1
done
echo "timeout"; exit 1
