"""Reads a GGUF and writes the LLM_CONFIG json that gen_llm.py, build.py and
the benchmark scripts consume. The "chat" block is derived from the embedded
chat template by chat_probe.py and verified against it (the printed OK/FAIL
lines); a FAIL or a probe error means fill or fix it by hand. Architecture quirks (sliding layers, NoPE layers, output gate, post norms,
embedding scale) must be checked against llama.cpp's src/models/<arch>.cpp.

usage: python inspect_gguf.py MODEL.gguf OUT.json   (needs `pip install gguf`)
"""
import json
import os
import sys

import gguf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import chat_probe  # noqa: E402

path, out = sys.argv[1], sys.argv[2]
r = gguf.GGUFReader(path)


def field(key, default=None):
    f = r.fields.get(key)
    if f is None:
        return default
    v = f.contents()
    return v


arch = field("general.architecture")
A = lambda k, d=None: field(f"{arch}.{k}", d)
tensors = {}
for t in r.tensors:
    tensors[t.name] = {"type": t.tensor_type.name.lower(), "shape": [int(x) for x in t.shape]}
H = A("embedding_length")
heads = A("attention.head_count")
kv = A("attention.head_count_kv")
head_dim = A("attention.key_length") or H // heads
L = A("block_count")
vocab = tensors["token_embd.weight"]["shape"][1]
template = field("tokenizer.chat_template", "") or ""
eos = field("tokenizer.ggml.eos_token_id")
tokens = field("tokenizer.ggml.tokens") or []
im_end = tokens.index("<|im_end|>") if "<|im_end|>" in tokens else None
endoftext = tokens.index("<|endoftext|>") if "<|endoftext|>" in tokens else None

cfg = {
    "name": field("general.name", arch),
    "arch": arch,
    "gguf": path,
    "hidden": H, "layers": L, "heads": heads, "kv_heads": kv if isinstance(kv, int) else max(kv),
    "head_dim": head_dim, "ffn": A("feed_forward_length"), "vocab": vocab,
    "context_length": A("context_length"),
    "rms_eps": A("attention.layer_norm_rms_epsilon", 1e-6),
    "rope_theta": A("rope.freq_base", 10000.0),
    # neox: rotary pairs (d, d + head_dim/2) -- qwen2/qwen3/gemma/...; norm: (2d, 2d+1) -- llama
    "rope_type": "norm" if arch in ("llama", "mistral", "granite") else "neox",
    "qk_norm": any(k.endswith(".attn_q_norm.weight") for k in tensors),
    "attn_gate": "blk.0.attn_gate.weight" in tensors,
    "post_norms": "blk.0.post_attention_norm.weight" in tensors,
    "embed_norm": False,
    "embed_scale": None,
    "sliding_window": A("attention.sliding_window") or 0,
    "sliding_layers": [],
    "nope_layers": [],
    "tied_embeddings": "output.weight" not in tensors,
    "eos_id": eos,
    # head_select stops on any of these; keep <|endoftext|> for ChatML models.
    "extra_eos_ids": sorted({i for i in (im_end, endoftext) if i is not None and i != eos}),
    "bos_id": field("tokenizer.ggml.bos_token_id") if field("tokenizer.ggml.add_bos_token", False) else None,
    "chat": None,
    "tensors": tensors,
}
# Partial rotary embedding (e.g. qwen35: 64 of 256 dims).
rope_dims = A("rope.dimension_count")
if rope_dims and rope_dims != head_dim:
    cfg["rope_dims"] = rope_dims
# Hybrid models: gated delta-net (linear attention) layers carry recurrent state
# instead of KV (llama.cpp qwen35/qwen3next: layer i is recurrent unless (i+1) % interval == 0).
if A("ssm.inner_size"):
    interval = A("full_attention_interval", 4)
    cfg["recurrent_layers"] = [i for i in range(L) if (i + 1) % interval != 0]
    cfg["gdn"] = {"conv_kernel": A("ssm.conv_kernel"), "k_heads": A("ssm.group_count"), "v_heads": A("ssm.time_step_rank"),
                  "head_k": A("ssm.state_size"), "inner": A("ssm.inner_size")}
    cfg["gdn"]["head_v"] = cfg["gdn"]["inner"] // cfg["gdn"]["v_heads"]
    cfg["kv_layers"] = L - len(cfg["recurrent_layers"])
    # qwen35-style attention: attn_q packs [query | sigmoid gate] per head; the FFN norm is post_attention_norm.
    attn0 = next(i for i in range(L) if i not in cfg["recurrent_layers"])
    if tensors[f"blk.{attn0}.attn_q.weight"]["shape"][1] == 2 * heads * head_dim:
        cfg["attn_gate"] = "fused_q"
    if f"blk.{attn0}.ffn_norm.weight" not in tensors and f"blk.{attn0}.post_attention_norm.weight" in tensors:
        cfg["post_norms"] = False
        cfg["ffn_norm_tensor"] = "post_attention_norm"
    print(f"hybrid: {len(cfg['recurrent_layers'])} gated delta-net layers + {cfg['kv_layers']} attention layers;", cfg["gdn"])
def generation_block(tpl):
    """Text the template emits for add_generation_prompt (what decides thinking mode)."""
    i = tpl.rfind("add_generation_prompt")
    return tpl[i:i + 300] if i >= 0 else ""


if template:
    try:
        bos_text = tokens[field("tokenizer.ggml.bos_token_id")] if field("tokenizer.ggml.bos_token_id") is not None else ""
        chat, notes = chat_probe.derive(template, bos_text)
        for n in notes:
            print("chat note:", n)
        results = chat_probe.verify(template, chat, bos_text)
        for name, ok, detail in results:
            print("chat", "OK  " if ok else "SKIP" if ok is None else "FAIL", name, "-", detail)
        cfg["chat"] = chat
        print(f"chat: thinking={chat['thinking']} (default {'on' if chat['thinking_default'] else 'off'}), "
              f"tool calls={chat['call_format']}; generation prompt {chat['generation_prompt']!r}")
        if any(ok is False for _, ok, _ in results):
            print("WARNING: the derived chat block does not reproduce the template (FAIL above); fix it by hand")
    except (chat_probe.ProbeError, ValueError, KeyError) as e:
        print("WARNING: chat template not derivable automatically:", e)
fmts = sorted({v["type"] for k, v in tensors.items() if k.endswith(".weight") and len(v["shape"]) == 2})
print(f"{cfg['name']}: arch={arch} L={L} H={H} heads={heads}/{cfg['kv_heads']} d={head_dim} ffn={cfg['ffn']} vocab={vocab} formats={fmts}")
unsupported = [f for f in fmts if f not in ("q4_k", "q5_k", "q6_k", "q8_0")
               and not (f == "f32" and cfg.get("gdn"))]  # f32 conv1d weights are read directly by the GDN kernels
if unsupported:
    print("WARNING: no kernels for", unsupported, "- add dequant helpers (see reference/porting-notes.md)")
if head_dim not in (128, 256):
    print("WARNING: head_dim", head_dim, "is not 128 or 256: the attention kernels support only those")
if cfg["chat"] is None:
    print("WARNING: no chat block; fill it by hand from tokenizer.chat_template (fields: see chat_probe.py)")
json.dump(cfg, open(out, "w"), indent=1)
print("wrote", out)
