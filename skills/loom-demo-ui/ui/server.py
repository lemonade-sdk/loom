# Loom demo UI: serves index.html and proxies /api/<target>/* to loom-serve instances.
# LOOM_UI_UPSTREAMS="gpu=http://127.0.0.1:8100,npu=http://127.0.0.1:8101" (default) names the targets;
# the page discovers each target's models (and their kinds) from /v1/status.
import os
import pathlib
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse, Response

UPSTREAM = dict(kv.split("=", 1) for kv in os.environ.get("LOOM_UI_UPSTREAMS", "gpu=http://127.0.0.1:8100,npu=http://127.0.0.1:8101").split(","))
HERE = pathlib.Path(__file__).parent
app = FastAPI()
client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=5.0))


@app.get("/")
async def index():
    return FileResponse(HERE / "index.html")


@app.get("/api/targets")
async def targets():
    return list(UPSTREAM)


@app.api_route("/api/{target}/{path:path}", methods=["GET", "POST", "DELETE"])
async def proxy(target: str, path: str, request: Request):
    if target not in UPSTREAM:
        return Response(status_code=404)
    headers = {k: v for k, v in request.headers.items() if k.lower() in ("content-type", "accept")}
    req = client.build_request(request.method, f"{UPSTREAM[target]}/{path}", content=await request.body(), headers=headers)
    try:
        upstream = await client.send(req, stream=True)
    except httpx.HTTPError as e:
        return Response(f'{{"error":{{"message":"{target} server unreachable: {type(e).__name__}"}}}}', status_code=502, media_type="application/json")

    async def body():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()
    return StreamingResponse(body(), status_code=upstream.status_code, media_type=upstream.headers.get("content-type"))


@app.get("/sample.wav")
async def sample():
    return FileResponse(HERE / "short.wav", media_type="audio/wav")
