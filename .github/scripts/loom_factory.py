"""Helpers for the loom-factory workflow.

  select   rank the trending GGUF chat LLMs from the approved publishers that fit a
           Strix Halo, and pick the first one that has no Loom package on the target org yet
  publish  upload a finished Loom package to the target org

Only `publish` needs huggingface_hub; `select` uses the HTTP API with the standard library.
"""
import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request

HF = "https://huggingface.co"
# Quant files the loom-port-llm starter has dequant helpers for, in order of preference.
QUANTS = ["Q4_K_M", "Q4_K_S", "Q5_K_M", "Q5_K_S", "Q6_K", "Q8_0"]
SHARD = re.compile(r"-\d{5}-of-\d{5}(?=\.gguf$)", re.I)
GIB = 2**30
# Hugging Face organizations whose GGUF repositories scheduled runs may convert: original model
# developers and established quantization providers (organization accounts, not individuals).
PUBLISHERS = [
    "unsloth", "ggml-org", "lmstudio-community",
    "Qwen", "google", "microsoft", "amd", "nvidia", "meta-llama", "mistralai", "deepseek-ai", "openai",
    "ibm-granite", "Intel", "tencent", "LiquidAI", "allenai", "HuggingFaceTB", "tiiuae", "CohereLabs",
    "LGAI-EXAONE", "zai-org", "openbmb", "internlm", "JetBrains",
]


def get(url, token=None):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"} if token else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        nxt = re.search(r'<([^>]+)>;\s*rel="next"', r.headers.get("Link", ""))
        return json.load(r), nxt.group(1) if nxt else None


def ranked_gguf_repos(publishers, token, per_publisher):
    """GGUF repos of `publishers`, highest Hugging Face trending score first."""
    repos = []
    for author in publishers:
        q = urllib.parse.urlencode([("author", author), ("filter", "gguf"), ("sort", "trendingScore"),
                                    ("direction", "-1"), ("limit", str(per_publisher)),
                                    ("expand[]", "trendingScore"), ("expand[]", "downloads"),
                                    ("expand[]", "gated"), ("expand[]", "private"), ("expand[]", "disabled"),
                                    ("expand[]", "gguf"), ("expand[]", "pipeline_tag")])
        repos += get(f"{HF}/api/models?{q}", token)[0]
    return sorted(repos, key=lambda m: -(m.get("trendingScore") or 0))


def pick_weights(repo, params, token):
    """The preferred quant of `repo` as (quant, [files], total_bytes), or None. Shards are summed."""
    files, url = [], f"{HF}/api/models/{repo}/tree/main?recursive=true"
    while url:
        page, url = get(url, token)
        files += [f for f in page if f.get("type") == "file" and f["path"].lower().endswith(".gguf")]
    groups = {}
    for f in files:
        name = os.path.basename(f["path"]).lower()
        # Vision projectors and speculative-decoding draft heads are not the model.
        if "mmproj" in name or "imatrix" in name or name.startswith("mtp") or "fastmtp" in name:
            continue
        groups.setdefault(SHARD.sub("", f["path"]), []).append(f)
    # Below ~3.2 bits/param the file can't hold the whole model at any of QUANTS.
    floor = 0.4 * (params or 0)
    for quant in QUANTS:
        pat = re.compile(rf"(^|[-_./]){quant}([-.]|$)", re.I)
        # Plain quants before vendor variants (e.g. unsloth's UD-Q4_K_M), then the smallest.
        hits = sorted(((not re.search(r"(^|[-_./])UD-", k, re.I), -sum(f["size"] for f in g), k, g)
                       for k, g in groups.items() if pat.search(k) and sum(f["size"] for f in g) >= floor),
                      reverse=True)
        if hits:
            _, neg, _, g = hits[0]
            return quant, sorted(f["path"] for f in g), -neg
    return None


def target_name(source, org):
    owner, name = source.split("/")
    name = re.sub(r"[-_.]?gguf$", "", name, flags=re.I)
    return f"{org}/{owner}-{name}-loom"[: len(org) + 1 + 96]


def published(org, token):
    """Repo ids on `org`, plus the source GGUF repo each one declares as base_model."""
    names, sources, url = set(), set(), f"{HF}/api/models?author={org}&limit=100&expand[]=cardData"
    while url:
        page, url = get(url, token)
        for m in page:
            names.add(m["id"])
            base = (m.get("cardData") or {}).get("base_model") or []
            sources.update([base] if isinstance(base, str) else base)
    return names, sources


def slug(repo):
    return repo.replace("/", "__")


def explicit(a, token):
    """The pick for a user-named checkpoint `org/repo` or `org/repo:path/in/repo.gguf`."""
    rid, _, path = a.checkpoint.strip().partition(":")
    q = urllib.parse.urlencode([("expand[]", "gguf"), ("expand[]", "gated"), ("expand[]", "downloads"),
                                ("expand[]", "trendingScore")])
    m, _ = get(f"{HF}/api/models/{rid}?{q}", token)
    g = m.get("gguf") or {}
    if path:
        files, url = [], f"{HF}/api/models/{rid}/tree/main?recursive=true"
        while url:
            page, url = get(url, token)
            files += [f for f in page if f.get("type") == "file"]
        group = [f for f in files if SHARD.sub("", f["path"]) == SHARD.sub("", path)]
        if not group:
            sys.exit(f"{path} not found in {rid}")
        quant = next((q for q in QUANTS if re.search(rf"(^|[-_./]){q}([-.]|$)", path, re.I)), "custom")
        w = quant, sorted(f["path"] for f in group), sum(f["size"] for f in group)
    else:
        w = pick_weights(rid, g.get("total"), token)
        if not w:
            sys.exit(f"{rid} has no {'/'.join(QUANTS)} file; name one with {rid}:<file.gguf>")
    if w[2] > a.max_gib * GIB:
        sys.exit(f"{rid} {w[0]} is {w[2] / GIB:.1f} GiB, over the {a.max_gib} GiB Strix Halo budget")
    if m.get("gated"):
        print(f"warning: {rid} is gated; the HF_LOOM_FACTORY account must have accepted its license")
    approved = rid.split("/")[0].lower() in {p.lower() for p in PUBLISHERS}
    names, sources = published(a.org, token)
    target = target_name(rid, a.org)
    note = "" if approved else ", publisher not on the approved list"
    return {"rank": 0, "repo": rid, "trending": m.get("trendingScore"), "downloads": m.get("downloads"),
            "architecture": g.get("architecture"),
            "params_b": round((g.get("total") or 0) / 1e9, 1), "quant": w[0], "files": w[1],
            "gib": round(w[2] / GIB, 1), "target": target,
            "status": f"todo (requested; already published, will update{note})" if target in names or rid in sources
            else f"todo (requested{note})"}


def cmd_select(a):
    token = os.environ.get("HF_TOKEN") or None
    if a.checkpoint:
        pick = explicit(a, token)
        json.dump({"org": a.org, "max_gib": a.max_gib, "top": [pick], "skipped": [], "pick": pick},
                  open(a.out, "w", encoding="utf-8"), indent=1)
        return report(a, [pick], [], pick, f"### Requested checkpoint (fits in {a.max_gib} GiB)")

    attempts = {}
    if a.attempts_file and os.path.exists(a.attempts_file):
        for line in open(a.attempts_file, encoding="utf-8"):
            m = re.match(r"loom-factory-attempt-(.+)-\d+$", line.strip())
            if m:
                attempts[m.group(1)] = attempts.get(m.group(1), 0) + 1
    names, sources = published(a.org, token)

    top, skipped = [], []
    for m in ranked_gguf_repos(PUBLISHERS, token, a.per_publisher):
        if len(top) >= a.top:
            break
        rid, g = m["id"], m.get("gguf") or {}
        why = None
        if m.get("private") or m.get("disabled"):
            why = "private/disabled"
        elif m.get("gated"):
            why = f"gated ({m['gated']})"
        elif not g.get("chat_template") or m.get("pipeline_tag") in ("feature-extraction", "sentence-similarity"):
            why = f"not a chat LLM (arch {g.get('architecture', '?')}, {m.get('pipeline_tag') or 'no pipeline tag'})"
        if not why:
            w = pick_weights(rid, g.get("total"), token)
            if not w:
                why = "no " + "/".join(QUANTS) + " file"
            elif w[2] > a.max_gib * GIB:
                why = f"{w[0]} is {w[2] / GIB:.1f} GiB > {a.max_gib} GiB"
        if why:
            skipped.append({"repo": rid, "trending": m.get("trendingScore"), "reason": why})
            continue
        target = target_name(rid, a.org)
        done = target in names or rid in sources
        tries = attempts.get(slug(rid), 0)
        top.append({"rank": len(top) + 1, "repo": rid, "trending": m.get("trendingScore"),
                    "downloads": m.get("downloads"),
                    "architecture": g.get("architecture"), "params_b": round((g.get("total") or 0) / 1e9, 1),
                    "quant": w[0], "files": w[1], "gib": round(w[2] / GIB, 1), "target": target,
                    "status": "published" if done else (f"failed {tries}x, skipped" if tries >= a.max_attempts
                                                        else "todo" + (f" (failed {tries}x)" if tries else ""))})
    pick = next((t for t in top if t["status"].startswith("todo")), None)
    result = {"org": a.org, "max_gib": a.max_gib, "top": top, "skipped": skipped, "pick": pick}
    json.dump(result, open(a.out, "w", encoding="utf-8"), indent=1)
    report(a, top, skipped, pick,
           f"### Top {len(top)} GGUF chat LLMs from the {len(PUBLISHERS)} approved publishers "
           f"(by Hugging Face trending score) that fit in {a.max_gib} GiB")


def report(a, top, skipped, pick, title):
    lines = [title, "", "| # | Source repo | Trending | Downloads (30d) | Arch | Params | Quant | Size | Status |",
             "|---|---|---|---|---|---|---|---|---|"]
    for t in top:
        mark = " **← this run**" if pick and t["repo"] == pick["repo"] else ""
        lines.append(f"| {t['rank'] or '-'} | [{t['repo']}]({HF}/{t['repo']}) | {t['trending'] or 0} | "
                     f"{t['downloads'] or 0:,} | "
                     f"{t['architecture']} | {t['params_b']}B | {t['quant']} | {t['gib']} GiB | {t['status']}{mark} |")
    if skipped:
        lines += ["", f"<details><summary>{len(skipped)} higher-ranked GGUF repos skipped</summary>", "",
                  *[f"- {s['repo']} (trending {s['trending'] or 0}): {s['reason']}" for s in skipped], "", "</details>"]
    lines += ["", f"Selected: `{pick['repo']}` ({', '.join(pick['files'])}) → `{pick['target']}`" if pick
              else "Nothing to do: every model in the list is published or out of attempts."]
    summary = "\n".join(lines) + "\n"
    sys.stdout.write(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8").write(summary)
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"found={'true' if pick else 'false'}\n")
            if pick:
                f.write(f"source={pick['repo']}\ntarget={pick['target']}\nslug={slug(pick['repo'])}\n"
                        f"quant={pick['quant']}\nfiles={json.dumps(pick['files'])}\ngib={pick['gib']}\n")


def cmd_publish(a):
    from huggingface_hub import HfApi, ModelCard

    api = HfApi(token=os.environ["HF_TOKEN"])
    pkg = os.path.join(a.work, "pkg")
    for need in ("model.json", "program"):
        if not os.path.exists(os.path.join(pkg, need)):
            sys.exit(f"package incomplete: {need} missing in {pkg}")
    if not any(f.endswith(".loom") for _, _, fs in os.walk(pkg) for f in fs):
        sys.exit(f"no .loom files in {pkg}")
    src = api.model_info(a.source, expand=["cardData"])
    license_ = (src.card_data or {}).get("license") if src.card_data else None
    files = json.loads(a.files)
    dl = " ".join(files)

    def read(name, parse=False):
        p = os.path.join(a.work, name)
        if not os.path.isfile(p):
            return None
        with open(p, encoding="utf-8") as f:
            return json.load(f) if parse else f.read()

    attempt, result = read("attempt.json", True) or {}, read("result.json", True) or {}
    info = {"source": a.source, "quant": a.quant, "files": files, "run": attempt.get("run"),
            "metrics": result.get("metrics"), "report": read("REPORT.md")}
    card = ModelCard(f"""---
base_model: {a.source}
{f"license: {license_}" if license_ else ""}
tags: [loom, loom-serve, gguf, strix-halo]
---

# {a.source.split('/')[1]} for AMD Loom

A [Loom](https://github.com/ROCm/hrx-system) package (`.loom` programs and a `model.json` card) for the
unmodified GGUF checkpoint [{a.source}]({HF}/{a.source}) (`{a.quant}`), generated on an AMD Ryzen AI Max+ 395
"Strix Halo" by the [loom-skills](https://github.com/{os.environ.get('GITHUB_REPOSITORY', 'loom-fun/loom-skills')})
factory workflow.

The weights are not included. Put them in `parameter/` before serving:

```bash
hf download {a.target} --local-dir pkg
hf download {a.source} {dl} --local-dir pkg/parameter
loom-serve --model=pkg/model.json --device=amdgpu --memory_bytes=... --listen=127.0.0.1:8080
```
""")
    api.create_repo(a.target, repo_type="model", private=a.private, exist_ok=True)
    with open(os.path.join(pkg, "README.md"), "w", encoding="utf-8") as f:
        f.write(str(card))
    # Performance numbers are not claims yet: they live only here, and the card must not mention them.
    with open(os.path.join(pkg, "info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
        f.write("\n")
    api.upload_folder(repo_id=a.target, folder_path=pkg, ignore_patterns=["parameter/*", "*.gguf"],
                      delete_patterns=["REPORT.md", "bench/**"],
                      commit_message=f"Loom package for {a.source} ({a.quant})")
    print(f"published {HF}/{a.target}")


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
s = sub.add_parser("select")
s.add_argument("--org", default="loom-fun")
s.add_argument("--top", type=int, default=20)
s.add_argument("--max-gib", type=float, default=48)
s.add_argument("--per-publisher", type=int, default=50, help="most-downloaded GGUF repos to consider per publisher")
s.add_argument("--max-attempts", type=int, default=2)
s.add_argument("--attempts-file", help="artifact names of previous attempts, one per line")
s.add_argument("--checkpoint", help="convert this org/repo[:file.gguf] instead of ranking")
s.add_argument("--out", default="selection.json")
p = sub.add_parser("publish")
p.add_argument("--work", required=True, help="directory with pkg/, attempt.json, result.json and REPORT.md")
p.add_argument("--source", required=True)
p.add_argument("--target", required=True)
p.add_argument("--quant", required=True)
p.add_argument("--files", required=True, help="JSON list of the GGUF file paths in the source repo")
p.add_argument("--private", action="store_true")
a = ap.parse_args()
{"select": cmd_select, "publish": cmd_publish}[a.cmd](a)
