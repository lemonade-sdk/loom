"""Reads a GGUF and writes the LLM_CONFIG json that gen_llm.py, build.py and
the benchmark scripts consume. Review the output by hand: the "chat" block is
only a guess (ChatML is detected from the embedded chat template), and
architecture quirks (sliding layers, NoPE layers, output gate, post norms,
embedding scale) must be checked against llama.cpp's src/models/<arch>.cpp.

usage: python inspect_gguf.py MODEL.gguf OUT.json   (needs `pip install gguf`)
"""
import json
import sys

import gguf

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
    "rms_eps": A("attention.layer_norm_rms_epsilon", 1e-6),
    "rope_theta": A("rope.freq_base", 10000.0),
    # neox: rotary pairs (d, d + head_dim/2) -- qwen2/qwen3/gemma/...; norm: (2d, 2d+1) -- llama
    "rope_type": "norm" if arch in ("llama", "mistral", "granite") else "neox",
    "qk_norm": "blk.0.attn_q_norm.weight" in tensors,
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
def generation_block(tpl):
    """Text the template emits for add_generation_prompt (what decides thinking mode)."""
    i = tpl.rfind("add_generation_prompt")
    return tpl[i:i + 300] if i >= 0 else ""


if "<|im_start|>" in template:
    gen_block = generation_block(template)
    # Thinking models open <think> in the generation prompt or let the model emit it
    # (Qwen3 hybrid: the template only *mentions* <think> for history). Instruct-only
    # variants (e.g. Qwen3-*-Instruct-2507) never think; their template still strips
    # <think> from history, so "<think>" anywhere in the template proves nothing.
    thinks = "<think>" in gen_block or ("enable_thinking" in template)
    cfg["chat"] = {
        "style": "chatml",
        "system_prefix": "<|im_start|>system\n", "user_prefix": "<|im_start|>user\n",
        "assistant_prefix": "<|im_start|>assistant\n", "suffix": "<|im_end|>\n",
        "think_open": "<think>\n", "think_close": "\n</think>\n\n",
        # Thinking models: generation starts inside the reasoning block.
        "generation_prompt": "<|im_start|>assistant\n<think>\n" if thinks else "<|im_start|>assistant\n",
        "starts_in_reasoning": thinks,
        "terminal": "<|im_end|>",
    }
    print("chat: ChatML,", "THINKING model (generation starts in <think>)" if thinks else "non-thinking (instruct) model",
          "- generation block of the template:\n   ", repr(gen_block[:200]))
    print("  CHECK this against the model card: hybrid Qwen3 thinks; *-Instruct-2507 / *-Thinking-2507 differ.")
fmts = sorted({v["type"] for k, v in tensors.items() if k.endswith(".weight") and len(v["shape"]) == 2})
print(f"{cfg['name']}: arch={arch} L={L} H={H} heads={heads}/{cfg['kv_heads']} d={head_dim} ffn={cfg['ffn']} vocab={vocab} formats={fmts}")
unsupported = [f for f in fmts if f not in ("q4_k", "q5_k", "q6_k", "q8_0")]
if unsupported:
    print("WARNING: no kernels for", unsupported, "- add dequant helpers (see reference/porting-notes.md)")
gemm_q5 = sorted({k for k, v in tensors.items() if k.startswith("blk.") and v["type"] == "q5_k"})
if gemm_q5:
    print("WARNING: q5_k layer tensors", gemm_q5[:4], "... have decode kernels but no prefill GEMM (gen_gemm.py:"
          " q4_k/q6_k/q8_0). Add a q5_k dequant16 case to gen_gemm.py first; q5_k is fine for the output head.")
if head_dim != 128:
    print("WARNING: head_dim", head_dim, "!= 128: attention kernels assume 128")
if cfg["chat"] is None:
    print("WARNING: chat template not ChatML; fill the 'chat' block by hand from tokenizer.chat_template")
json.dump(cfg, open(out, "w"), indent=1)
print("wrote", out)
