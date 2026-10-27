---
name: loom-setup
description: Install AMD Loom (hrx-loom wheel from ROCm/hrx-system CI) and the hrx-loom-library catalog from a cold start, fetch catalog models with loom-fetch, and start/stop/inspect loom-serve on the GPU or NPU. Use for "install loom", "loom-fetch a model", "start loom-serve for X", "is the server ready", NPU serving.
---

# Loom setup: install → fetch → serve

Read the `loom` umbrella skill's hard rules first (memory limits, `--memory_bytes`,
binary approval).

## 1. Install (≈15 s with a warm pip cache, a few minutes cold)

```bash
bash ${CLAUDE_SKILL_DIR}/scripts/install_loom.sh ~/loom            # wheel pinned by the library (sha256-checked)
bash ${CLAUDE_SKILL_DIR}/scripts/install_loom.sh ~/loom latest     # or newest main CI build
```
The script:
- clones `ROCm/hrx-loom-library` to `~/loom/library`;
- downloads the `loom-sdk-wheel-linux-Release` CI artifact with `gh api .../artifacts/<id>/zip`
  (`gh run download` fails with a path-traversal error on these zips);
- creates `~/loom/.venv` with `pip install 'hrx_loom-…whl[hf]'`, which provides loom-serve,
  loom-fetch, loom-test, loom-run, loom-compile, loom-benchmark, loom-opt, loom-format, etc.;
- creates `~/loom/.tools`, a venv with gguf, numpy, tokenizers, requests, fastapi, uvicorn
  and httpx for the port, benchmark and UI scripts.

Facts:
- There is no PyPI release. CI artifacts expire after about 90 days. If the pinned one is
  gone, use `latest` and note which commit you got (`gh api repos/ROCm/hrx-system/actions/artifacts/<id>`).
- You need `gh auth status` to be logged in. `python3 -m venv` needs Python ≥ 3.12.
- The library pins its SDK in `library/build_tools/ci/sdk.json` (revision, run_id,
  artifact_id, sha256). A wheel newer than the pin usually works; an older one may
  not compile newer library programs.
- **NPU:** the Qwen3-Embedding NPU program compiled only with the wheel from
  hrx-system PR 2259's CI (artifact 12623044407 at the time). Install it separately:
  `VENV_NAME=.venv-npu install_loom.sh ~/loom 12623044407`. If that artifact has
  expired, look for a newer PR/main build whose NPU path compiles.

## 2. Fetch a catalog model

```bash
cd ~/loom && . .venv/bin/activate
L=library/model
loom-fetch $L/moonshine-ai/moonshine-streaming-tiny/model.json --list            # presets / variants (use --dry-run for sizes)
loom-fetch $L/moonshine-ai/moonshine-streaming-tiny/model.json --preset=streaming --output=models/moonshine --no-input
loom-fetch $L/Qwen/Qwen3.8-27B/model.json --preset=vision --output=models/qwen27b-vision --no-input --dry-run   # size check first
```
Notes:
- `--dry-run` prints what would be downloaded. Use it to check sizes against the
  memory budget.
- Weights come from Hugging Face; the HF cache (`~/.cache/huggingface/hub`) is reused.
  Gated repos need the user to accept the license on the HF page first.
- If fetch complains about a card/lock mismatch for a library model, add
  `--origin=library=$PWD/library` so program sources resolve against your checkout.
- Without a checkout you can fetch remote cards directly:
  `loom-fetch https://raw.githubusercontent.com/ROCm/hrx-loom-library/main/model/<org>/<name>/model.json --preset=… --output=…`
- Catalog models seen working on Strix Halo:

  | Model | Kind | Preset |
  |---|---|---|
  | Qwen3.8-27B | text + vision | `vision` |
  | moonshine-streaming-tiny | ASR (`recognition`) | `streaming` |
  | kyutai/pocket-tts | TTS (`speech`) | `streaming` |
  | krea/Krea-2-Turbo | image | `image` or `image-int8` |
  | Qwen3-Embedding-0.6B | embedding (NPU) | `embedding` |

  **Qwen3.8-Flash-Next (104 GiB) does not fit. Never serve it here.**

## 3. Serve

```bash
S=${CLAUDE_SKILL_DIR}/scripts
VENV=~/loom/.venv $S/serve.sh asr 8100 amdgpu 16000000000 ~/loom/models/moonshine/model.json
VENV=~/loom/.venv $S/serve.sh gpu 8100 amdgpu 53687091200 models/a/model.json models/b/model.json -- --parameter_idle_ms=60000
VENV=~/loom/.venv-npu $S/serve.sh npu 8101 'xdna://0?columns=8' 4000000000 models/qwen3-embedding/model.json -- --pool_backing=fixed
$S/stop.sh asr
```
`serve.sh` waits until `/v1/status` reports every model `registration: ready` and
checks that its PID owns the port. If it reports `failed`, it prints the compile
diagnostics. Logs, PIDs and ports go in `~/.cache/loom-serve/NAME.{log,pid,port}`.
Use unique NAMEs. `stop.sh` refuses to kill a PID that is not a loom-serve.

Raw form: `loom-serve --device=amdgpu --listen=127.0.0.1:PORT --model=pkg/model.json
--workers=8 --memory_bytes=N [--preset=NAME] [--parameter_idle_ms=60000]`.

## 4. Use / inspect

```bash
curl -s localhost:8100/v1/status | python3 -m json.tool      # per model: id, loom.kind, registration, compilation, parameters, status
curl -s localhost:8100/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<id from status>","messages":[{"role":"user","content":"hi"}],"max_tokens":64}'
```
Endpoints by `loom.kind`:

| Kind | Endpoint | Request |
|---|---|---|
| `text` | `/v1/chat/completions` | Supports streaming and `stream_options.include_usage`. Reasoning arrives in `reasoning_content`. |
| `recognition` | `/v1/audio/transcriptions` | Multipart, `audio/pcm;rate=16000;channels=1`. |
| `speech` | `/v1/audio/speech` | Needs `input` and `voice`. `{"response_format":"pcm"}` gives 24 kHz s16le. |
| `image` | `/v1/images/generations` | `{"response_format":"b64_json","size":"512x512"}`. |
| `embedding` | `/v1/embeddings` | |

Transcription example. The server takes raw 16 kHz mono s16le PCM:
```bash
ffmpeg -loglevel error -i in.wav -ar 16000 -ac 1 -f s16le /tmp/x.pcm
curl -s localhost:8100/v1/audio/transcriptions -F model=moonshine-ai/moonshine-streaming-tiny \
  -F "file=@/tmp/x.pcm;type=audio/pcm;rate=16000;channels=1"            # -> {"text": "..."}
```
TTS: `curl -s localhost:8100/v1/audio/speech -H 'Content-Type: application/json' -d '{"model":"kyutai/pocket-tts","input":"Hello.","voice":"alba","response_format":"pcm"}' > out.pcm`
returns 24 kHz s16le mono. `voice` is required. Voices are the card's `voice/*.wav` names
(alba, marius, anna, …). `seed` is optional.

The `model` field is the card's `model.id`, not a file name. Unknown models return
404, and unsupported fields are rejected rather than ignored.

Server behavior:
- Startup compiles the program, typically 10–60 s on a first run (cached afterwards).
- The first request loads weights lazily, so exclude it from timing.
- Special tokens are stripped from decoded output text. A control program that must
  see structural tokens in the output needs them marked non-special in the
  package's `tokenizer.json`; see `loom-port-llm`.
- Upstream docs: `loom/docs/src/serving/workflows/{fetch-and-serve,serve-models,operate-model-server,observe-model-server}.md`
  in hrx-system.
