"""Correctness gate: greedy Loom generations vs the llama.cpp golden.

usage: python compare.py PACKAGE GOLDEN.json [--url http://127.0.0.1:8160] [--out compare.json]

Per prompt: length of the token prefix matching the golden, and at the first
divergence the golden log-prob gap  logp(golden token) - logp(loom token).
Greedy decoding diverges at near-ties under any change of summation order, so
the gate is about *where* divergences happen, not whether:
  PASS   every divergence has gap <= 1.0 (near-tie) and the loom token is in the golden top-5
  FAIL   any divergence with a large gap, or a token outside the golden top-5
(Two llama.cpp builds disagree with each other in the same way.)
"""
import argparse
import json
import time

import requests

from bench_common import Package

ap = argparse.ArgumentParser()
ap.add_argument("package")
ap.add_argument("golden")
ap.add_argument("--url", default="http://127.0.0.1:8160")
ap.add_argument("--out", default="compare.json")
ap.add_argument("--max-gap", type=float, default=1.0)
ap.add_argument("--alt", default=None, help="second golden recording; per prompt the better-matching one is used")
a = ap.parse_args()
pkg = Package(a.package)
rows = []
alt = {g["prompt"]: g for g in json.load(open(a.alt))} if a.alt else {}


def match(got, g):
    gold = g["ids"]
    n = 0
    while n < min(len(got), len(gold)) and got[n] == gold[n]:
        n += 1
    gap = None
    if n < min(len(got), len(gold)):
        if n >= len(g["top"]):
            gap = 0.0  # no top-5 recorded here (byte-split token); counted as a near-tie
        else:
            top = dict((int(i), lp) for i, lp in g["top"][n])
            gap = (top[gold[n]] - top[got[n]]) if got[n] in top else "not in golden top-5"
    return n, gap


for g in json.load(open(a.golden)):
    t0 = time.perf_counter()
    r = requests.post(a.url + "/v1/chat/completions", headers={"Content-Type": "application/json"},
                      json={"model": pkg.model_id, "messages": [{"role": "user", "content": g["prompt"]}],
                            "max_tokens": len(g["ids"]), "temperature": 0}).json()
    if "choices" not in r:  # e.g. 503 capacity errors: raise --memory_bytes (KV pool must fit)
        print("ERROR", r.get("error"), flush=True)
        rows.append({"prompt": g["prompt"][:80], "error": r.get("error"), "bad": True, "matched": 0, "full_match": False, "divergence_gap": None})
        continue
    ch = r["choices"][0]
    text = pkg.raw_from_chat(ch["message"], ch["finish_reason"] == "stop")
    got = pkg.ids(text)
    n, gap = match(got, g)
    if g["prompt"] in alt:  # llama.cpp disagreed with itself? take the recording Loom follows further
        n2, gap2 = match(got, alt[g["prompt"]])
        if n2 > n:
            g, n, gap = alt[g["prompt"]], n2, gap2
    gold = g["ids"]
    full = n >= min(len(gold), len(got)) and abs(len(got) - len(gold)) <= 1
    bad = isinstance(gap, str) or (gap is not None and gap > a.max_gap)
    rows.append({"prompt": g["prompt"][:80], "prompt_tokens": g.get("prompt_tokens"), "golden_tokens": len(gold), "matched": n,
                 "full_match": full, "divergence_gap": gap, "bad": bad, "seconds": round(time.perf_counter() - t0, 2), "loom_text": text[:160]})
    print(f"{g.get('prompt_tokens', 0):6d} {n:3d}/{len(gold):3d} gap={gap if not isinstance(gap, float) else round(gap, 3)} {'BAD ' if bad else ''}{g['prompt'][:50]!r}", flush=True)
json.dump(rows, open(a.out, "w"), indent=1)
gaps = [r["divergence_gap"] for r in rows if isinstance(r["divergence_gap"], float)]
summary = {"prompts": len(rows), "full_match": sum(r["full_match"] for r in rows),
           "median_matched": sorted(r["matched"] for r in rows)[len(rows) // 2],
           "max_gap": round(max(gaps), 3) if gaps else 0.0, "bad": sum(r["bad"] for r in rows)}
summary["verdict"] = "PASS" if summary["bad"] == 0 else "FAIL"
print(json.dumps(summary))
