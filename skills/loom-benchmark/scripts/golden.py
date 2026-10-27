"""Golden reference: greedy llama.cpp generations (token ids + top-5 log-probs)
for the short prompt set and (optionally) long real-text prompts.

usage: python golden.py PACKAGE OUT.json [--llama http://127.0.0.1:8130] [--tokens 96] [--long 1000,2000,16000,32000]

llama.cpp gets the *rendered prompt as raw token ids* (/completion), so it sees
exactly the tokens Loom's control program produces. Run llama-server with the
same GGUF the Loom package uses. llama.cpp ROCm is not bit-deterministic across
runs at near-ties: record a second golden (OUT2.json) and pass it to compare.py
with --alt so a prompt where llama.cpp disagrees with itself is not called a FAIL.
"""
import argparse
import json

import requests

from bench_common import PROMPTS, Package, doc

ap = argparse.ArgumentParser()
ap.add_argument("package")
ap.add_argument("out")
ap.add_argument("--llama", default="http://127.0.0.1:8130")
ap.add_argument("--tokens", type=int, default=96)
ap.add_argument("--long", default="", help="comma-separated doc sizes for long-context goldens (64 tokens each)")
a = ap.parse_args()
pkg = Package(a.package)
items = [(p, a.tokens) for p in PROMPTS]
for n in [int(x) for x in a.long.split(",") if x]:
    for q in ("Summarize this documentation in one sentence.", "What is the main command shown in this documentation?"):
        items.append((q + "\n\n" + doc(n), 64))
out = []
for prompt, n_predict in items:
    ids = pkg.ids(pkg.render(prompt))
    r = requests.post(a.llama + "/completion", json={"prompt": ids, "n_predict": n_predict, "temperature": 0, "n_probs": 5,
                                                     "cache_prompt": False, "return_tokens": True}).json()
    gen = r["tokens"]  # completion_probabilities omits byte-fallback (partial UTF-8) tokens, e.g. emoji
    cp = r["completion_probabilities"]
    if len(cp) != len(gen):  # keep top-5 lists only while they line up with the ids
        k = next((i for i, x in enumerate(cp) if i >= len(gen) or x["id"] != gen[i]), len(cp))
        cp = cp[:k]
    out.append({"prompt": prompt, "prompt_tokens": len(ids), "ids": gen, "text": r["content"],
                "top": [[(t["id"], t["logprob"]) for t in x["top_logprobs"]] for x in cp]})
    print(f"{len(ids):6d} prompt tokens -> {len(gen):3d} {r['content'][:60]!r}", flush=True)
json.dump(out, open(a.out, "w"))
print("wrote", a.out)
