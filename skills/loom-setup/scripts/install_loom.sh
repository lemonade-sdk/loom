#!/bin/bash
# Install the Loom SDK (hrx-loom) + the matching model library, from scratch.
# usage: [VENV_NAME=.venv] [PYTHON=python3] install_loom.sh [ROOT=~/loom] [VERSION|latest|ci-latest|ci:ARTIFACT_ID]
#   0.1.0 (default):  hrx-loom from PyPI at the version the loom-port-llm starter is validated with
#   latest:           newest hrx-loom on PyPI (the starter's templates may need updating for a new ABI)
#   ci-latest:        newest "loom-sdk-wheel-linux-Release" artifact from ROCm/hrx-system main (development builds;
#                     Windows: "loom-sdk-wheel-windows"); needs an authenticated gh
#   ci:ARTIFACT_ID:   that CI artifact
# The library (ROCm/hrx-loom-library) is checked out at the catalog revision the installed SDK names in
# hrx_loom-*.dist-info/loom.json, so catalog cards and the SDK agree.
# Needs: python >= 3.12 with venv, git (and gh + unzip for ci-*). Windows: run from Git Bash (venvs use Scripts/).
set -euo pipefail
root=${1:-$HOME/loom}; which=${2:-0.1.0}
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) win=1; bin=Scripts; py=${PYTHON:-py -3};; *) win=0; bin=bin; py=${PYTHON:-python3};; esac
mkdir -p "$root"; cd "$root"
venv=${VENV_NAME:-.venv}
$py -m venv $venv
$venv/$bin/python -m pip install -q --upgrade pip
newest() {  # newest unexpired artifact named $1, preferring main and v* tags
  gh api "repos/ROCm/hrx-system/actions/artifacts?name=$1&per_page=50" \
    -q '[.artifacts[] | select(.expired | not)] | (map(select(.workflow_run.head_branch == "main" or ((.workflow_run.head_branch // "") | startswith("v")))) + .) | .[0].id'
}
case "$which" in
  ci-latest|ci:*)
    if [ "$which" = ci-latest ]; then
      if [ $win = 1 ]; then art=$(newest loom-sdk-wheel-windows); else art=$(newest loom-sdk-wheel-linux-Release); fi
    else art=${which#ci:}; fi
    [ -n "$art" ] && [ "$art" != null ] || { echo "no SDK wheel artifact found"; exit 1; }
    echo "artifact $art ($(gh api repos/ROCm/hrx-system/actions/artifacts/$art -q '.workflow_run.head_branch + " " + .workflow_run.head_sha[:12]'))"
    mkdir -p wheel-$art
    # `gh run download` trips a path-traversal check on these zips; fetch the zip directly.
    [ -f wheel-$art/sdk.zip ] || gh api repos/ROCm/hrx-system/actions/artifacts/$art/zip > wheel-$art/sdk.zip
    unzip -o -q wheel-$art/sdk.zip -d wheel-$art
    # Some artifacts bundle +asan/+debug/+tracy variants next to the release wheel; take the plain one.
    whl=$(ls wheel-$art/*.whl | grep -v '+' | head -1)
    $venv/$bin/pip install -q "$whl[hf]";;
  latest) $venv/$bin/pip install -q --upgrade 'hrx-loom[hf]';;
  *) $venv/$bin/pip install -q "hrx-loom[hf]==$which";;
esac
# The SDK names its compatible catalog as "github:ROCm/hrx-loom-library@<revision>".
catalog=$($venv/$bin/python - <<'EOF'
import glob, json, os, site
for d in site.getsitepackages():
    for f in glob.glob(os.path.join(d, "hrx_loom-*.dist-info", "loom.json")):
        print(json.load(open(f)).get("catalog", "").rpartition("@")[2]); raise SystemExit
EOF
)
[ -d library/.git ] || git clone -q https://github.com/ROCm/hrx-loom-library library
if [ -n "$catalog" ]; then
  git -C library fetch -q origin "$catalog" 2>/dev/null || git -C library fetch -q origin
  git -C library checkout -q "$catalog" && echo "library at catalog revision ${catalog:0:12}"
else
  echo "SDK names no catalog revision (development build); library left at $(git -C library rev-parse --short HEAD)"
fi
# Tools venv for ports/benchmarks/UI (kept separate from the SDK venv).
[ -d .tools ] || $py -m venv .tools
.tools/$bin/pip install -q gguf numpy tokenizers requests jinja2 fastapi uvicorn httpx
$venv/$bin/loom-serve --help >/dev/null && echo "OK: hrx-loom $($venv/$bin/python -c 'import importlib.metadata as m;print(m.version("hrx-loom"))') in $root/$venv/$bin; tools venv $root/.tools"
