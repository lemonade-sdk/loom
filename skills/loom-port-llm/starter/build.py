"""Builds a loom-serve package for a dense GGUF LLM from its LLM_CONFIG json.

usage: python build.py CONFIG.json PACKAGE_DIR --tokenizer TOKENIZER.json
           [--context 32768] [--pool 65536] [--prefill 1024] [--rows 2] [--max-tokens 512]

Writes PACKAGE_DIR/{model.json, weights.loom, tokenizer.json, program/*.loom,
parameter/<gguf> (symlink)}. Serve it with
  loom-serve --model=PACKAGE_DIR/model.json --device=amdgpu --memory_bytes=... --listen=...
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("config")
ap.add_argument("package")
ap.add_argument("--tokenizer", required=True, help="HF tokenizer.json of the same model (download from the base repo)")
ap.add_argument("--context", type=int, default=32768)
ap.add_argument("--pool", type=int, default=65536)
ap.add_argument("--prefill", type=int, default=1024, help="prefill chunk (prefill_capacity); 1024-4096")
ap.add_argument("--rows", type=int, default=2, help="concurrent sequences (rows); c4 benchmarks need >= 4")
ap.add_argument("--max-tokens", type=int, default=512)
ap.add_argument("--model-id", default=None)
a = ap.parse_args()

cfg = json.load(open(a.config))
chat = cfg.get("chat")
if not chat:
    sys.exit("config has no 'chat' block: fill it from the model's chat template first")
pkg = os.path.abspath(a.package)
os.makedirs(os.path.join(pkg, "program"), exist_ok=True)
os.makedirs(os.path.join(pkg, "parameter"), exist_ok=True)

# 1. Kernel library + command program.
gen = os.path.join(pkg, "gen")
env = dict(os.environ, LLM_CONFIG=os.path.abspath(a.config), LLM_OUT=gen)
subprocess.run([sys.executable, os.path.join(HERE, "gen_llm.py"), os.path.abspath(a.config)], check=True, env=env, stdout=subprocess.DEVNULL)
os.replace(os.path.join(gen, "model.loom"), os.path.join(pkg, "program", "llm.loom"))

# 2. Serving glue from templates.
hx = lambda s: s.encode().hex()
pos_bytes = cfg["kv_heads"] * cfg["head_dim"] * 2
subs = {
    "@@POS_BYTES@@": str(pos_bytes), "@@PLANES@@": str(2 * cfg["layers"]), "@@BLOCK_BYTES@@": str(16 * pos_bytes),
    "@@MAX_CONTEXT@@": str(a.context), "@@TERMINAL@@": chat["terminal"], "@@HEX_TERMINAL@@": hx(chat["terminal"]),
    "@@HEX_UNSUPPORTED@@": hx(f"llm requires pooled epochs without MTP, at most 128 shapes and a context of at most {a.context}"),
    "@@HEX_NAME@@": hx(cfg.get("name", "llm")),
    "@@HEX_SYSTEM_PREFIX@@": hx(chat["system_prefix"]), "@@HEX_USER_PREFIX@@": hx(chat["user_prefix"]),
    "@@HEX_ASSISTANT_PREFIX@@": hx(chat["assistant_prefix"]), "@@HEX_SUFFIX@@": hx(chat["suffix"]),
    "@@HEX_THINK_OPEN@@": hx(chat.get("think_open", "")), "@@HEX_THINK_CLOSE@@": hx(chat.get("think_close", "")),
    "@@HEX_GENERATION_PROMPT@@": hx(chat["generation_prompt"]),
    "@@START_STATE@@": "1" if chat.get("starts_in_reasoning") else "0",
    "@@HEX_THINK_END@@": hx(chat.get("think_end", "</think>")),
    "@@CHAT_COMMENT@@": json.dumps({k: v for k, v in chat.items()})[:400],
}
for name in ("prepare", "control", "storage", "service"):
    text = open(os.path.join(HERE, "templates", name + ".loom")).read()
    for k, v in subs.items():
        text = text.replace(k, v)
    assert "@@" not in text, (name, [l for l in text.splitlines() if "@@" in l])
    open(os.path.join(pkg, "program", name + ".loom"), "w").write(text)
open(os.path.join(pkg, "weights.loom"), "w").write(open(os.path.join(HERE, "templates", "weights.loom")).read())

# 3. Tokenizer. loom-serve strips *special* added tokens from decoded text; any
# structural token the control program must see in the output (channel tags,
# e.g. Glimmer's <|start|>/<|message|>) is listed in chat.nonspecial_tokens.
tok = json.load(open(a.tokenizer))
keep = set(chat.get("nonspecial_tokens", []))
for t in tok.get("added_tokens", []):
    if t["content"] in keep:
        t["special"] = False
json.dump(tok, open(os.path.join(pkg, "tokenizer.json"), "w"), ensure_ascii=False)

# 4. Weights (symlink) and the package card.
gguf_name = os.path.basename(cfg["gguf"])
link = os.path.join(pkg, "parameter", gguf_name)
if os.path.lexists(link):
    os.remove(link)
os.symlink(os.path.abspath(cfg["gguf"]), link)
card = {
    "schema_version": 1,
    "model": {"id": a.model_id or cfg.get("name", "llm"), "name": f"{cfg.get('name', 'llm')} (Loom port, GGUF)"},
    "programs": {"source": {
        "inputs": [{"path": "program/prepare.loom", "data": ["weights.loom"]}, "program/control.loom", "program/storage.loom", "program/service.loom"],
        "libraries": ["program/llm.loom"]}},
    "services": {"chat": {"kind": "text", "version": 1}},
    "variants": {"gguf": {"services": {"chat": {
        "programs": ["source"], "config": {"llm.card": True},
        "parameters": {"model": {"files": [{"path": f"parameter/{gguf_name}", "format": "gguf"}]}},
        "tokenizers": {"text": "tokenizer.json"}}}}},
    "presets": {"chat": {"service": "chat", "variant": "gguf", "programs": "source", "execution": {
        "rows": a.rows, "context_capacity": a.context, "pool_capacity": a.pool,
        "prefill_capacity": a.prefill, "max_tokens": a.max_tokens}}},
}
json.dump(card, open(os.path.join(pkg, "model.json"), "w"), indent=1)
# The benchmark scripts read the chat format and model id from here.
bench = {k: v for k, v in cfg.items() if k != "tensors"}
bench["model_id"] = card["model"]["id"]
json.dump(bench, open(os.path.join(pkg, "llm_config.json"), "w"), indent=1)
kv = a.pool * pos_bytes * 2 * cfg["layers"]
weights = os.path.getsize(os.path.abspath(cfg["gguf"]))
need = weights + kv + 2 * 2**30
print("package:", os.path.join(pkg, "model.json"))
print(f"KV pool {kv / 2**30:.1f} GiB + weights {weights / 2**30:.1f} GiB: serve with --memory_bytes >= {need // 10**9 + 1}000000000"
      " (too small shows up as 503 'retained state leaves insufficient completion capacity')")
