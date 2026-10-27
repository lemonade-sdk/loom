#!/bin/bash
# Alternative to the plugin install: copy the skills into ~/.claude/skills so they
# load as plain user skills (/loom, /loom-setup, ...).
# usage: ./install.sh            (or: curl/clone the repo first)
#        ./install.sh --uninstall
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
dest=${CLAUDE_SKILLS_DIR:-$HOME/.claude/skills}
skills=(loom loom-setup loom-demo-ui loom-port-llm loom-roofline loom-optimize loom-benchmark)
if [ "${1:-}" = "--uninstall" ]; then
  for s in "${skills[@]}"; do rm -rf "$dest/$s"; done; echo "removed ${skills[*]} from $dest"; exit 0
fi
mkdir -p "$dest"
for s in "${skills[@]}"; do
  rm -rf "$dest/$s"; cp -a "$here/skills/$s" "$dest/$s"
done
echo "installed ${#skills[@]} skills into $dest - start a new Claude Code session and run /loom"
