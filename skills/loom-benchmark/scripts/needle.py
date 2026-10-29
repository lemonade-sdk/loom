"""Full-context gate: needle-in-a-haystack retrieval up to the package's context capacity.

"Access code" sentences (one by default, at a depth that varies with the size; --needles 3
puts them at 10/50/90%) are hidden in a real-text haystack (the skill's docs, then the local
Python stdlib sources) and the model is asked to list them. Sizes default to 25/50/75/100% of the package's context_capacity (minus the answer
budget). It also checks that a prompt over the capacity is refused cleanly (HTTP 400)
and that the server is still healthy afterwards.

usage: needle.py PKG LOOM_URL [--sizes N ...] [--probes 2] [--needles 1|3] [--llama URL] [--max-tokens 1024] [--out FILE]
  default sizes: 50 % and 100 % of context_capacity (a full-context prefill can take many minutes)
  --llama also runs the identical prompt ids through llama.cpp /completion for comparison
  (start it with a single slot whose context covers the largest size).
Each size runs --probes haystacks (different needles and depths). Small models miss needles
on their own, and near the context limit both engines flip between found and missed on a
near-tie, so the gate is aggregate: Loom must find at least half of all probes, and with
--llama at least as many as llama.cpp minus one. Exit status 1 on a failed gate or a failed
over-limit check. Answers are saved with --out.
"""
import argparse
import glob
import json
import os
import random
import sys
import sysconfig
import time

import requests

from bench_common import DATA, Package

QUESTIONS = {
    1: ("\n\n----\nSomewhere in the material above there is exactly one sentence of the form "
        "'The access code for vault <NAME> is <CODE>.' What is the vault's name and code? Answer as NAME: CODE."),
    3: ("\n\n----\nSomewhere in the material above there are exactly three sentences of the form "
        "'The access code for vault <NAME> is <CODE>.' List all three vaults with their codes, "
        "one per line as NAME: CODE. Answer only with those three lines."),
}


def corpus():
    parts = [open(os.path.join(DATA, f"doc_{n}.txt"), encoding="utf-8").read() for n in (32000, 16000, 8000)]
    stdlib = sysconfig.get_paths()["stdlib"]
    for f in sorted(glob.glob(os.path.join(stdlib, "*.py"))):
        parts.append(f"\n\n# ===== file: {os.path.basename(f)} =====\n" + open(f, encoding="utf-8", errors="replace").read())
    return "".join(parts)


def haystack(P, ids, n, seed, count=1):
    rng = random.Random(seed)
    body = P.tok.decode(ids[:n])
    names = rng.sample(["ORCHID", "TUNDRA", "MERIDIAN", "BASALT", "KESTREL", "QUARTZ", "LANTERN", "SAFFRON"], count)
    codes = [f"{rng.randint(100, 999)}-{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.randint(10, 99)}" for _ in names]
    depths = (0.1, 0.5, 0.9) if count == 3 else (rng.choice((0.1, 0.3, 0.5, 0.7, 0.9)),)
    for depth, name, code in sorted(zip(depths, names, codes), reverse=True):
        cut = body.rfind("\n", 0, int(len(body) * depth)) + 1
        body = body[:cut] + f"The access code for vault {name} is {code}.\n" + body[cut:]
    return body + QUESTIONS[count], dict(zip(names, codes))


def chat(url, model, prompt, max_tokens, extra):
    t0 = time.time(); ttft = None; text = ""; usage = None; finish = None
    r = requests.post(url + "/v1/chat/completions", stream=True, timeout=7200, json={
        "model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
        "temperature": 0, "stream": True, "stream_options": {"include_usage": True}, **extra})
    if r.status_code != 200:
        return {"status": r.status_code, "error": r.text[:300]}
    for line in r.iter_lines():
        if not line.startswith(b"data: {"):
            continue
        d = json.loads(line[6:])
        usage = d.get("usage") or usage
        for c in d.get("choices", []):
            delta = c.get("delta", {})
            piece = (delta.get("reasoning_content") or "") + (delta.get("content") or "")
            if piece and ttft is None:
                ttft = time.time() - t0
            text += piece
            finish = c.get("finish_reason") or finish
    return {"status": 200, "text": text, "ttft_s": ttft, "total_s": time.time() - t0, "usage": usage, "finish": finish}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pkg")
    ap.add_argument("url")
    ap.add_argument("--sizes", type=int, nargs="*")
    ap.add_argument("--llama", default=None)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--needles", type=int, default=1, choices=[1, 3])
    ap.add_argument("--probes", type=int, default=2)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    P = Package(a.pkg)
    card = json.load(open(os.path.join(a.pkg, "model.json")))
    context = next(iter(card["presets"].values()))["execution"]["context_capacity"]
    # Thinking off (when the template allows it) keeps answers short and deterministic.
    thinking = False if P.chat.get("thinking") == "toggle" else None
    extra = {"chat_template_kwargs": {"enable_thinking": False}} if thinking is False else {}
    overhead = len(P.ids(P.render_messages([{"role": "user", "content": QUESTIONS[a.needles]}], None, thinking)))
    top = context - a.max_tokens - overhead - 64
    sizes = a.sizes or [top // 2, top]
    ids = P.ids(corpus())
    while len(ids) < top:
        ids = ids + ids  # small stdlib: repeat (needles stay unique)
    print(f"context {context}; haystack sizes {sizes}", flush=True)
    failures = 0
    results = []
    loom_total = llama_total = probes_total = 0
    for n in sizes:
        for probe in range(a.probes):
            prompt, needles = haystack(P, ids, n, n * 31 + probe, a.needles)
            rec = {"n": n, "probe": probe}
            r = chat(a.url, P.model_id, prompt, a.max_tokens, extra)
            rec.update(prompt_tokens=(r.get("usage") or {}).get("prompt_tokens"), status=r["status"], ttft_s=r.get("ttft_s"),
                       finish=r.get("finish"), found=sum(v in r.get("text", "") for v in needles.values()), error=r.get("error"),
                       needles=needles, answer=r.get("text", "")[-400:])
            if a.llama:
                pids = P.ids(P.render_messages([{"role": "user", "content": prompt}], None, thinking))
                t0 = time.time()
                g = requests.post(a.llama + "/completion", timeout=7200, json={
                    "prompt": pids, "n_predict": a.max_tokens, "temperature": 0, "cache_prompt": False}).json()
                rec.update(llama_found=sum(v in g.get("content", "") for v in needles.values()), llama_s=round(time.time() - t0, 1),
                           llama_error=(g.get("error") or {}).get("message"), llama_answer=g.get("content", "")[-400:])
                llama_total += rec["llama_found"]
            failures += rec["status"] != 200
            loom_total += rec["found"]; probes_total += a.needles
            print(("OK  " if rec["status"] == 200 else "FAIL"), json.dumps({k: v for k, v in rec.items() if "answer" not in k}), flush=True)
            results.append(rec)
    enough = loom_total * 2 >= probes_total and (not a.llama or loom_total >= llama_total - 1)
    failures += not enough
    print(("OK  " if enough else "FAIL"), f"needles found: Loom {loom_total}/{probes_total}" + (f", llama.cpp {llama_total}/{probes_total}" if a.llama else ""))
    # Over the capacity: a clean refusal, then the server still answers.
    prompt, _ = haystack(P, ids * 2, context + 1000, 1)
    r = requests.post(a.url + "/v1/chat/completions", timeout=600, json={
        "model": P.model_id, "messages": [{"role": "user", "content": prompt}], "max_tokens": 16, **extra})
    healthy = requests.post(a.url + "/v1/chat/completions", timeout=600, json={
        "model": P.model_id, "messages": [{"role": "user", "content": "Say OK."}], "max_tokens": 64, **extra}).status_code == 200
    over_ok = r.status_code == 400 and healthy
    failures += not over_ok
    print(("OK  " if over_ok else "FAIL"), f"over-capacity prompt -> HTTP {r.status_code} {r.text[:160]!r}; healthy afterwards: {healthy}")
    if a.out:
        json.dump({"context": context, "results": results, "over_capacity": {"status": r.status_code, "healthy": healthy}},
                  open(a.out, "w"), indent=1)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
