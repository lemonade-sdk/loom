#!/bin/bash
# Stop a server started by serve.sh, by its PID file. Refuses to kill a PID that is
# not a loom-serve (stale PID file). Never `pkill -f` a pattern: it can match your own shell.
name=$1; state=${LOOM_SERVE_STATE:-$HOME/.cache/loom-serve}; f=$state/$name.pid
[ -f "$f" ] || { echo "no $f"; exit 1; }
pid=$(cat "$f")
if ! tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | grep -q loom-serve; then echo "pid $pid is not a loom-serve (stale $f); removing it"; rm -f "$f"; exit 1; fi
kill "$pid"; for i in $(seq 30); do kill -0 "$pid" 2>/dev/null || { echo "stopped $name"; rm -f "$f"; exit 0; }; sleep 0.5; done
kill -9 "$pid" 2>/dev/null; rm -f "$f"; echo "killed $name"
