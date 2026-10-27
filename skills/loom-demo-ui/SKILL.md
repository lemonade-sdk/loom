---
name: loom-demo-ui
description: Start and use the Loom Playground web UI (chat with streaming reasoning and tok/s, ASR with mic recording, TTS, image generation, embedding similarity) that proxies to one or more running loom-serve instances and discovers their models automatically. Use when the user wants a browser demo/UI for Loom models.
---

# Loom Playground UI

A small FastAPI proxy (`ui/server.py`) + one static page (`ui/index.html`). The page
polls each upstream's `/v1/status`, groups ready models by `loom.kind`, and wires
the tabs:

| Tab | Model kind | What it does |
|---|---|---|
| Chat | `text` | Dropdown of every text model. Streams the reply, shows reasoning first, then reports TTFT and decode tok/s. Image attach is allowed when the model lists `image` input. |
| Transcribe | `recognition` | Upload a file, record from the mic, or use the bundled `short.wav`. The browser converts audio to 16 kHz mono PCM. |
| Speak | `speech` | Plays the result and reports real-time factor. |
| Image | `image` | 512² or 1024², with a seed. |
| Embed | `embedding` | One sentence per line; shows a cosine-similarity matrix. |

The first served model of each kind is used, and missing kinds show "(none)".

## Start

1. Start the model servers with `loom-setup` (`serve.sh`), for example a GPU server on
   :8100 with several models and an NPU server on :8101.
2. Start the UI. The tools venv needs fastapi, uvicorn and httpx:
   ```bash
   LOOM_UI_UPSTREAMS="gpu=http://127.0.0.1:8100,npu=http://127.0.0.1:8101" \
     bash ${CLAUDE_SKILL_DIR}/start_ui.sh 8090 ~/loom/.tools
   ```
   It prints `UI: http://127.0.0.1:8090`. The log is `/tmp/loom-ui-8090.log`; to stop it,
   run `kill $(cat /tmp/loom-ui-8090.pid)`.
3. Open the page on the same machine. The microphone needs `localhost` or https.
   Remotely, use an SSH tunnel: `ssh -L 8090:127.0.0.1:8090 host`.

`LOOM_UI_UPSTREAMS` takes any number of `name=url` pairs, and the name becomes the
route `/api/<name>/…`. You can add models without touching the UI: restart the
upstream loom-serve with another `--model=`, and within 5 s the page picks it up from
`/v1/status`.

## Verify (headless)

```bash
curl -s localhost:8090/api/targets                       # ["gpu","npu"]
curl -s localhost:8090/api/gpu/v1/status | head -c 200
chromium --headless --no-sandbox --virtual-time-budget=8000 --dump-dom http://127.0.0.1:8090/ \
  | grep -o '<select id="chatmodel">.*</select>\|data-t="[a-z]*"[^<]*'   # dropdown + tab labels populated
```
Always pipe `--dump-dom` as shown. Snap chromium's stdout redirected straight to a
file gives 0 bytes, under `/tmp` and `$HOME` alike. If you need a file, use `… --dump-dom URL | cat > ~/dom.html`.

## Recommended demo set on Strix Halo

The GPU server shares a 50 GiB elastic budget (`--memory_bytes=53687091200
--parameter_idle_ms=60000`) across:
- Qwen3.8-27B vision (preset `vision`)
- moonshine-streaming-tiny
- pocket-tts
- Krea-2-Turbo `image-int8`
- optionally your own ported LLM package

The NPU server runs Qwen3-Embedding-0.6B (older wheel, `--pool_backing=fixed`).

Switching between big models takes a few seconds while weights reload.
