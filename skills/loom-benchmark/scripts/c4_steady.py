"""Steady-state aggregate decode throughput with N concurrent streams: tokens
produced by all streams inside the window where every stream is decoding
(excludes ramp-up and tail, unlike perf.py's c4_agg_tps).

usage: python c4_steady.py URL [N=4] [--model MODEL_ID]   (omit --model for llama.cpp)
Loom needs execution.rows >= N in the package preset; llama-server needs -np N.
"""
import argparse
import json
import threading
import time

import requests

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("n", type=int, nargs="?", default=4)
ap.add_argument("--model", default=None)
ap.add_argument("--max-tokens", type=int, default=400)
a = ap.parse_args()
events = [[] for _ in range(a.n)]


def run(i):
    body = {"messages": [{"role": "user", "content": f"[{time.time()} {i}] Write a very long, detailed essay about the history of lighthouses."}],
            "max_tokens": a.max_tokens, "stream": True, "temperature": 0}
    if a.model:
        body["model"] = a.model
    with requests.post(a.url + "/v1/chat/completions", json=body, stream=True, headers={"Content-Type": "application/json"}) as r:
        for line in r.iter_lines():
            if not line.startswith(b"data: {"):
                continue
            for c in json.loads(line[6:]).get("choices", []):
                dl = c.get("delta", {})
                if dl.get("content") or dl.get("reasoning_content"):
                    events[i].append(time.perf_counter())


warm = {"messages": [{"role": "user", "content": "Say hi."}], "max_tokens": 8, "temperature": 0}
if a.model:
    warm["model"] = a.model
requests.post(a.url + "/v1/chat/completions", json=warm, headers={"Content-Type": "application/json"})  # warm-up
ts = [threading.Thread(target=run, args=(i,)) for i in range(a.n)]
[t.start() for t in ts]
[t.join() for t in ts]
start = max(e[0] for e in events)
end = min(e[-1] for e in events)
count = sum(sum(1 for x in e if start <= x <= end) for e in events)
print(json.dumps({"streams": a.n, "window_s": round(end - start, 2), "agg_tps": round(count / (end - start), 1),
                  "per_stream_events": [len(e) for e in events]}))
