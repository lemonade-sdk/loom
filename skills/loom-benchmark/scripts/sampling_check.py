"""Sampling gate: temperature / top_p / top_k / seed behave as specified.

Checks, on one short prompt with thinking off when the template allows it:
  1. temperature 0 == a request without sampling fields (greedy), and top_k 1 == greedy;
  2. the same seed reproduces the same completion; different seeds vary;
  3. the empirical first-token distribution at temperature 1 (N draws, distinct seeds) matches
     llama.cpp's probabilities for the same prompt ids (total-variation distance), when --llama is given.

usage: sampling_check.py PKG LOOM_URL [--llama URL] [--draws 200]
"""
import argparse
import collections
import json
import math
import sys

import requests

from bench_common import Package

PROMPT = "Pick a random fruit: apple, banana, cherry or grape. Answer with just the fruit name in lowercase."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkg")
    ap.add_argument("url")
    ap.add_argument("--llama", default=None)
    ap.add_argument("--draws", type=int, default=200)
    a = ap.parse_args()
    P = Package(a.pkg)
    thinking = False if P.chat.get("thinking") == "toggle" else None
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if thinking is False else {}
    msgs = [{"role": "user", "content": PROMPT}]

    def ask(max_tokens=16, **kw):
        r = requests.post(a.url + "/v1/chat/completions", timeout=600,
                          json={"model": P.model_id, "messages": msgs, "max_tokens": max_tokens, **extra, **kw}).json()
        if "error" in r:
            raise SystemExit(f"Loom error: {r['error']}")
        m = r["choices"][0]["message"]
        return (m.get("reasoning_content") or "") + (m.get("content") or "")

    fails = 0
    greedy = ask()
    checks = [("temperature 0 == greedy", ask(temperature=0) == greedy),
              ("top_k 1 == greedy", ask(temperature=1.0, top_k=1, seed=7) == greedy)]
    same = [ask(temperature=1.0, seed=1234) for _ in range(2)]
    checks.append(("same seed reproduces", same[0] == same[1]))
    varied = {ask(temperature=1.5, seed=s) for s in range(12)}
    checks.append((f"different seeds vary ({len(varied)} distinct of 12)", len(varied) > 1))
    for name, ok in checks:
        fails += not ok
        print(("OK  " if ok else "FAIL"), name)
    if a.llama:
        first = collections.Counter()
        for s in range(a.draws):
            first[P.ids(ask(max_tokens=1, temperature=1.0, top_p=1.0, top_k=0, seed=1000 + s))[0]] += 1
        ids = P.ids(P.render_messages(msgs, None, thinking))
        g = requests.post(a.llama + "/completion", json={"prompt": ids, "n_predict": 1, "temperature": 0, "n_probs": 20,
                                                          "cache_prompt": False}).json()
        probs = {t["id"]: math.exp(t["logprob"]) for t in g["completion_probabilities"][0]["top_logprobs"]}
        support = set(probs) | set(first)
        tv = 0.5 * sum(abs(probs.get(t, 0.0) - first[t] / a.draws) for t in support)
        # Sampling noise alone gives TV ~ sqrt(k / N); allow 3x that plus a small numerical margin.
        bound = 3 * math.sqrt(max(2, len([p for p in probs.values() if p > 0.01])) / a.draws) / 2 + 0.05
        ok = tv <= bound
        fails += not ok
        top = sorted(support, key=lambda t: -probs.get(t, 0))[:5]
        print(("OK  " if ok else "FAIL"), f"first-token distribution: total variation {tv:.3f} (bound {bound:.3f}); "
              + ", ".join(f"{P.tok.id_to_token(t)!r} llama {probs.get(t, 0):.2f} loom {first[t] / a.draws:.2f}" for t in top))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
