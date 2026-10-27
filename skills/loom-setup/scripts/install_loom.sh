#!/bin/bash
# Install the Loom SDK (hrx-loom wheel) + the model library, from scratch.
# usage: [VENV_NAME=.venv] install_loom.sh [ROOT=~/loom] [pinned|latest|ARTIFACT_ID]
#   pinned (default): the exact wheel the library's build_tools/ci/sdk.json pins (sha256-checked)
#   latest:           newest "loom-sdk-wheel-linux-Release" artifact from ROCm/hrx-system main CI
# Needs: gh (authenticated: `gh auth status`), python3 >= 3.12 with venv, unzip, git.
# There is no PyPI release; the wheel is a GitHub Actions artifact (expires after ~90 days).
set -euo pipefail
root=${1:-$HOME/loom}; which=${2:-pinned}
mkdir -p "$root"; cd "$root"
[ -d library ] || gh repo clone ROCm/hrx-loom-library library -- --depth 50
pin=library/build_tools/ci/sdk.json
case "$which" in
  pinned) art=$(python3 -c "import json;print(json.load(open('$pin'))['artifact_id'])"); sha=$(python3 -c "import json;print(json.load(open('$pin'))['sha256'])");;
  latest) run=$(gh run list -R ROCm/hrx-system --workflow ci_iree.yml --branch main --status success --limit 1 --json databaseId -q '.[0].databaseId')
          art=$(gh api repos/ROCm/hrx-system/actions/runs/$run/artifacts -q '.artifacts[] | select(.name=="loom-sdk-wheel-linux-Release") | .id'); sha="";;
  *) art=$which; sha="";;
esac
echo "artifact $art"
mkdir -p wheel-$art
# `gh run download` trips a path-traversal check on these zips; fetch the zip directly.
[ -f wheel-$art/sdk.zip ] || gh api repos/ROCm/hrx-system/actions/artifacts/$art/zip > wheel-$art/sdk.zip
unzip -o -q wheel-$art/sdk.zip -d wheel-$art
whl=$(ls wheel-$art/*.whl | head -1)
if [ -n "$sha" ]; then echo "$sha  $whl" | sha256sum -c -; fi
venv=${VENV_NAME:-.venv}
python3 -m venv $venv
$venv/bin/pip install -q --upgrade pip
$venv/bin/pip install -q "$whl[hf]"
# Tools venv for ports/benchmarks/UI (kept separate from the SDK venv).
[ -d .tools ] || python3 -m venv .tools
.tools/bin/pip install -q gguf numpy tokenizers requests fastapi uvicorn httpx
$venv/bin/loom-serve --help >/dev/null && echo "OK: loom tools in $root/$venv/bin; tools venv $root/.tools"
