"""Batching correctness: prompts run concurrently must give the same greedy
output as the same prompts run alone. Concurrent prefill packs several
sequences into one epoch, so 16-token attention tiles straddle sequences and
exercise the mixed/fallback paths that single requests never hit.

usage: python consistency.py URL --model MODEL_ID [--n 4]
"""
import argparse
import concurrent.futures as cf

import requests

from bench_common import doc

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("--model", required=True)
ap.add_argument("--n", type=int, default=4)
a = ap.parse_args()
body = doc(1000)
prompts = [f"Question {i}: summarize part {i} of this text.\n\n" + body[i * 700:(i * 700) + 700 + 157 * i] for i in range(a.n)]


def run(p):
    r = requests.post(a.url + "/v1/chat/completions", headers={"Content-Type": "application/json"},
                      json={"model": a.model, "messages": [{"role": "user", "content": p}], "max_tokens": 40, "temperature": 0}).json()
    m = r["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + "|" + (m.get("content") or ""), r["usage"]["prompt_tokens"]


alone = [run(p) for p in prompts]
with cf.ThreadPoolExecutor(a.n) as ex:
    together = list(ex.map(run, prompts))
same = [x[0] == y[0] for x, y in zip(alone, together)]
print("prompt tokens", [x[1] for x in alone], "identical when concurrent:", same)
print("PASS" if all(same) else "FAIL")
from campaign import log_gate
log_gate("consistency", "PASS" if all(same) else "FAIL", identical=same)
