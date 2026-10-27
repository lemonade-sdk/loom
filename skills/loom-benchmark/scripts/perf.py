"""Throughput on fixed workloads, for Loom or llama.cpp (same prompts both sides).

usage: python perf.py URL LABEL [--kind loom|llama] [--model MODEL_ID] [--long] [--log perf_log.jsonl]

Metrics (all greedy, real-text documents, each prompt starts with a nonce so no
prefix cache can help):
  decode_tps        300-token essay from a short prompt
  prefill[]         ~1k and ~2k-token documents, max_tokens=4: prompt_tokens / TTFT,
                    median (and min/max) of --repeats runs; single samples vary up to 2x
  decode_tps_1k/2k  256-token generation after a 1k / 2k document (decode with KV in use)
  c4_agg_tps        8 requests of 200 tokens through 4 client threads: total tokens / wall time
                    (includes ramp-up/tail; c4_steady.py measures the steady state)
  --long: prefill 8k/16k/32k documents and decode_tps_32k (128 tokens after 32k)
Loom numbers are client-side (streaming TTFT and token arrival). llama.cpp
numbers come from its server-side `timings` (excludes HTTP and tokenization),
which slightly favours llama.cpp.
"""
import argparse
import concurrent.futures as cf
import json
import random
import time

import requests

from bench_common import doc, stream

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("label")
ap.add_argument("--kind", default="loom", choices=["loom", "llama"])
ap.add_argument("--model", default=None, help="Loom model id (model.json model.id)")
ap.add_argument("--long", action="store_true")
ap.add_argument("--log", default="perf_log.jsonl")
ap.add_argument("--reverted", action="store_true", help="this build is not kept (still logged: its time counts)")
ap.add_argument("--repeats", type=int, default=5, help="samples per short-prefill size (median reported)")
a = ap.parse_args()
U, M = a.url, a.model
H = {"Content-Type": "application/json"}


def run(content, max_tokens):
    """-> (prompt_tokens, prefill_tps, decode_tps, completion_tokens)"""
    msgs = [{"role": "user", "content": content}]
    if a.kind == "loom":
        s = stream(U, M, msgs, max_tokens)
        return s["prompt_tokens"], s["prompt_tokens"] / s["ttft"], s["decode_tps"], s["tokens"]
    d = requests.post(U + "/v1/chat/completions", headers=H, json={"messages": msgs, "max_tokens": max_tokens, "temperature": 0, "cache_prompt": False}).json()
    t = d["timings"]
    return t["prompt_n"], t["prompt_per_second"], t["predicted_per_second"], d["usage"]["completion_tokens"]


run("Say hi.", 8)  # warm-up: the first request after a restart loads weights lazily
out = {"label": a.label, "kind": a.kind, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
nonce = str(time.time())
if a.long:
    out["prefill_long"] = []
    for n in (8000, 16000, 32000):
        pt, pps, _, _ = run(f"[{nonce} {n}] Summarize this documentation in one sentence.\n\n" + doc(n), 4)
        out["prefill_long"].append({"prompt_tokens": pt, "prefill_tps": round(pps, 1)})
        print(out["prefill_long"][-1], flush=True)
    _, _, dtps, _ = run(f"[{nonce} ctx32] Write a long, detailed essay discussing this documentation.\n\n" + doc(32000), 128)
    out["decode_tps_32k"] = round(dtps, 2)
else:
    _, _, out["decode_tps"], _ = run("Write a detailed 300-word essay about the history of the printing press.", 300)
    out["prefill"] = []
    for n in (1000, 2000):  # short prefills are noisy (scheduling, clocks): median of a.repeats fresh-nonce runs
        samples = [run(f"[{nonce} {i} {random.random()}] Summarize this documentation in one sentence.\n\n" + doc(n), 4) for i in range(a.repeats)]
        pps = sorted(x[1] for x in samples)
        out["prefill"].append({"prompt_tokens": samples[0][0], "prefill_tps": round(pps[len(pps) // 2], 1),
                               "min": round(pps[0], 1), "max": round(pps[-1], 1), "n": len(pps)})
    for n in (1000, 2000):
        _, _, out[f"decode_tps_{n // 1000}k"], _ = run(f"[{nonce} ctx] Write a long, detailed essay discussing this documentation.\n\n" + doc(n), 256)

    def one(i):
        return run(f"[{nonce}] Write a short story ({i}) about a lighthouse keeper, about 200 words.", 200)[3]
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(4) as ex:
        toks = list(ex.map(one, range(8)))
    out["c4_agg_tps"] = round(sum(toks) / (time.perf_counter() - t0), 1)
for k, v in list(out.items()):
    if isinstance(v, float):
        out[k] = round(v, 2)
print(json.dumps(out))
with open(a.log, "a") as f:
    f.write(json.dumps(out) + "\n")
if a.kind == "loom":  # campaign tracking: targets.jsonl + rate-of-approach verdict (needs roofline.json)
    from campaign import log_metrics
    vals = {k: out.get(k) for k in ("decode_tps", "decode_tps_1k", "decode_tps_2k", "decode_tps_32k", "c4_agg_tps")}
    for p in out.get("prefill", []) + out.get("prefill_long", []):
        n = min((1000, 2000, 8000, 16000, 32000), key=lambda x: abs(x - p["prompt_tokens"]))
        vals[f"prefill_{n}"] = p["prefill_tps"]
    log_metrics(vals, a.label, a.reverted, a.log)
