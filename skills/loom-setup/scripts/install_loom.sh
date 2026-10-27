#!/bin/bash
# Install the Loom SDK (hrx-loom wheel) + the model library, from scratch.
# usage: [VENV_NAME=.venv] [PYTHON=python3] install_loom.sh [ROOT=~/loom] [pinned|latest|ARTIFACT_ID]
#   pinned (default): the exact wheel the library's build_tools/ci/sdk.json pins (sha256-checked; Linux only)
#   latest:           newest "loom-sdk-wheel-linux-Release" artifact from ROCm/hrx-system main
#                     (Windows: newest "loom-sdk-wheel-windows", preferring main / v* release tags)
# Needs: gh (authenticated: `gh auth status`), python >= 3.12 with venv, unzip, git.
# Windows: run from Git Bash; venv executables are in .venv/Scripts instead of .venv/bin.
# There is no PyPI release; the wheel is a GitHub Actions artifact (expires after ~90 days).
set -euo pipefail
root=${1:-$HOME/loom}; which=${2:-pinned}
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) win=1; bin=Scripts; py=${PYTHON:-py -3};; *) win=0; bin=bin; py=${PYTHON:-python3};; esac
mkdir -p "$root"; cd "$root"
[ -d library ] || gh repo clone ROCm/hrx-loom-library library -- --depth 50
pin=library/build_tools/ci/sdk.json
# The library pins a Linux wheel only.
if [ "$which" = pinned ] && [ $win = 1 ]; then echo "no pinned Windows wheel; using latest"; which=latest; fi
newest() {  # newest unexpired artifact named $1, preferring main and v* tags
  gh api "repos/ROCm/hrx-system/actions/artifacts?name=$1&per_page=50" \
    -q '[.artifacts[] | select(.expired | not)] | (map(select(.workflow_run.head_branch == "main" or ((.workflow_run.head_branch // "") | startswith("v")))) + .) | .[0].id'
}
case "$which" in
  pinned) art=$($py -c "import json;print(json.load(open('$pin'))['artifact_id'])"); sha=$($py -c "import json;print(json.load(open('$pin'))['sha256'])");;
  latest) if [ $win = 1 ]; then art=$(newest loom-sdk-wheel-windows); else art=$(newest loom-sdk-wheel-linux-Release); fi; sha="";;
  *) art=$which; sha="";;
esac
[ -n "$art" ] && [ "$art" != null ] || { echo "no SDK wheel artifact found"; exit 1; }
echo "artifact $art ($(gh api repos/ROCm/hrx-system/actions/artifacts/$art -q '.workflow_run.head_branch + " " + .workflow_run.head_sha[:12]'))"
mkdir -p wheel-$art
# `gh run download` trips a path-traversal check on these zips; fetch the zip directly.
[ -f wheel-$art/sdk.zip ] || gh api repos/ROCm/hrx-system/actions/artifacts/$art/zip > wheel-$art/sdk.zip
unzip -o -q wheel-$art/sdk.zip -d wheel-$art
# Some artifacts bundle +asan/+debug/+tracy variants next to the release wheel; take the plain one.
whl=$(ls wheel-$art/*.whl | grep -v '+' | head -1)
if [ -n "$sha" ]; then echo "$sha  $whl" | sha256sum -c -; fi
venv=${VENV_NAME:-.venv}
$py -m venv $venv
$venv/$bin/python -m pip install -q --upgrade pip
$venv/$bin/pip install -q "$whl[hf]"
# Tools venv for ports/benchmarks/UI (kept separate from the SDK venv).
[ -d .tools ] || $py -m venv .tools
.tools/$bin/pip install -q gguf numpy tokenizers requests fastapi uvicorn httpx
$venv/$bin/loom-serve --help >/dev/null && echo "OK: loom tools in $root/$venv/$bin; tools venv $root/.tools"
