"""Kernel unit tests for a gen_llm.py build: numpy references from gguf dequantization.

usage: python make_kernel_tests.py CONFIG.json OUTDIR      (tools venv: gguf, numpy)
       run_kernel_tests.sh PACKAGE OUTDIR                  (loom-test on the GPU)
Covers g_embed, the decode matvecs (1- and 4-token variants) for one tensor of
each (format, K) used by layers 0-1, and the QK-norm + RoPE kernel. Extend it
when you add a kernel: a kernel that passes here but breaks end-to-end is
usually a launch-shape/indexing assumption (e.g. one-wave rows), so test the
real shapes of the model, not toy ones.
"""
import json, sys, os, numpy as np
from gguf import GGUFReader
from gguf.quants import dequantize
cfg = json.load(open(sys.argv[1])); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
r = GGUFReader(cfg["gguf"]); T = {t.name: t for t in r.tensors}
def raw(name): return np.asarray(T[name].data).view(np.int8).reshape(-1)
def deq(name):
    t = T[name]; return dequantize(np.asarray(t.data), t.tensor_type).reshape(int(t.shape[1]), int(t.shape[0])) if len(t.shape) == 2 else np.asarray(t.data, dtype=np.float32)
H, D, QH, KVH, eps = cfg["hidden"], cfg["head_dim"], cfg["heads"], cfg["kv_heads"], cfg["rms_eps"]
rng = np.random.default_rng(0)
cases = []
def save(n, a): np.save(os.path.join(out, n + ".npy"), a)
def ty(a): return "tensor<" + "x".join(map(str, a.shape)) + "x" + {"float32": "f32", "int8": "i8", "int32": "i32", "float16": "f16"}[str(a.dtype)] + ">"
# 1. embed
ids = np.array([0, cfg['vocab'] - 1, cfg['vocab'] // 2, 1000], dtype=np.int32); t = len(ids)
E = deq("token_embd.weight"); w = deq("blk.0.attn_norm.weight")
x = E[ids]; h = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w
tab = raw("token_embd.weight")
for n, a in [("e_ids", ids), ("e_tab", tab), ("e_w", w.astype(np.float32)), ("e_x", x.astype(np.float32)), ("e_h", h.astype(np.float32))]: save(n, a)
cases.append(f"""check.case public @embed {{
  %t = check.literal value({t}) : index
  %ids = check.file.read.npy path("e_ids.npy") : {ty(ids)}
  %tab = check.file.read.npy path("e_tab.npy") : {ty(tab)}
  %w = check.file.read.npy path("e_w.npy") : tensor<{H}xf32>
  %ex = check.file.read.npy path("e_x.npy") : tensor<{t}x{H}xf32>
  %eh = check.file.read.npy path("e_h.npy") : tensor<{t}x{H}xf32>
  %x = check.generate.fill value(0.0) : tensor<{t}x{H}xf32>
  %h = check.generate.fill value(0.0) : tensor<{t}x{H}xf32>
  kernel.launch @g_embed[%t](%t, %ids, %tab, %w, %x, %h) : [index](index, {ty(ids)}, {ty(tab)}, tensor<{H}xf32>, tensor<{t}x{H}xf32>, tensor<{t}x{H}xf32>)
  check.expect.close actual(%x) expected(%ex) atol(1.0e-4) rtol(1.0e-3) nan(different) : tensor<{t}x{H}xf32>
  check.expect.close actual(%h) expected(%eh) atol(1.0e-3) rtol(1.0e-3) nan(different) : tensor<{t}x{H}xf32>
  check.return
}}
""")
# 2. matvec per (tensor) for t=1 and t=4 variants
seen = set(); todo = []
for li in range(cfg["layers"]):
    for nm_, K in [("attn_q", H), ("attn_k", H), ("attn_v", H), ("ffn_down", cfg["ffn"]), ("attn_output", QH * D), ("ffn_gate", H)]:
        name = f"blk.{li}.{nm_}.weight"; key = (T[name].tensor_type.name.lower(), K)
        if key not in seen: seen.add(key); todo.append((name, K))
for name, K in todo:
    fmt = T[name].tensor_type.name.lower(); W = deq(name); n = W.shape[0]
    for tg in (1, 4):
        xs = rng.standard_normal((tg, K)).astype(np.float32); y = xs @ W.T.astype(np.float64)
        tag = name.replace(".", "_") + f"_t{tg}"
        save(tag + "_w", raw(name)); save(tag + "_x", xs); save(tag + "_y", y.astype(np.float32))
        kern = f"g_mv_{fmt}_k{K}" + ("_t4" if tg == 4 else "")
        wt = ty(raw(name))
        cases.append(f"""check.case public @mv_{tag} {{
  %n = check.literal value({n}) : index
  %t = check.literal value({tg}) : index
  %acc = check.literal value(0) : i32
  %w = check.file.read.npy path("{tag}_w.npy") : {wt}
  %x = check.file.read.npy path("{tag}_x.npy") : tensor<{tg}x{K}xf32>
  %e = check.file.read.npy path("{tag}_y.npy") : tensor<{tg}x{n}xf32>
  %y = check.generate.fill value(0.0) : tensor<{tg}x{n}xf32>
  kernel.launch @{kern}[%n, %t](%n, %t, %acc, %x, %w, %y) : [index, index](index, index, i32, tensor<{tg}x{K}xf32>, {wt}, tensor<{tg}x{n}xf32>)
  check.expect.close actual(%y) expected(%e) atol(2.0e-3) rtol(2.0e-3) nan(different) : tensor<{tg}x{n}xf32>
  check.return
}}
""")
# 3. qk norm + rope on q (live = 0: no KV store)
t = 4; pos = np.array([0, 1, 7, 300])
qn = deq("blk.0.attn_q_norm.weight") if cfg["qk_norm"] else np.ones(D, np.float32)
kn = deq("blk.0.attn_k_norm.weight") if cfg["qk_norm"] else np.ones(D, np.float32)
q = rng.standard_normal((t, QH, D)).astype(np.float32); k = rng.standard_normal((t, KVH, D)).astype(np.float32); v = rng.standard_normal((t, KVH, D)).astype(np.float32)
def norm(a, w): return a / np.sqrt((a * a).mean(-1, keepdims=True) + eps) * w
def rope(a):
    inv = cfg["rope_theta"] ** (-np.arange(0, D // 2) * 2.0 / D); ang = pos[:, None] * inv[None]
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    if cfg["rope_type"] == "neox":
        a1, a2 = a[..., :D // 2], a[..., D // 2:]; return np.concatenate([a1 * c - a2 * s, a2 * c + a1 * s], -1)
    a1, a2 = a[..., 0::2], a[..., 1::2]; o = np.empty_like(a); o[..., 0::2] = a1 * c - a2 * s; o[..., 1::2] = a2 * c + a1 * s; return o
qe = rope(norm(q, qn) if cfg["qk_norm"] else q)
info = np.zeros((t, 4), np.int32); info[:, 1] = pos
for n, a in [("r_q", q.reshape(t, -1)), ("r_k", k.reshape(t, -1)), ("r_v", v.reshape(t, -1)), ("r_qn", qn.astype(np.float32)), ("r_kn", kn.astype(np.float32)), ("r_qe", qe.reshape(t, -1).astype(np.float32)), ("r_info", info), ("r_orig", np.zeros(64, np.int32))]: save(n, a)
cases.append(f"""check.case public @qk_rope {{
  %t = check.literal value({t}) : index
  %layer = check.literal value(0) : index
  %rope = check.literal value(1) : i32
  %ctx = check.literal value(1024) : index
  %pool = check.literal value(1024) : index
  %q = check.file.read.npy path("r_q.npy") : tensor<{t}x{QH * D}xf32>
  %k = check.file.read.npy path("r_k.npy") : tensor<{t}x{KVH * D}xf32>
  %v = check.file.read.npy path("r_v.npy") : tensor<{t}x{KVH * D}xf32>
  %qn = check.file.read.npy path("r_qn.npy") : tensor<128xf32>
  %kn = check.file.read.npy path("r_kn.npy") : tensor<128xf32>
  %info = check.file.read.npy path("r_info.npy") : tensor<{t}x4xi32>
  %orig = check.file.read.npy path("r_orig.npy") : tensor<64xi32>
  %qe = check.file.read.npy path("r_qe.npy") : tensor<{t}x{QH * D}xf32>
  %arena = check.generate.fill value(0.0) : tensor<1024xf16>
  kernel.launch @g_qk_rope_store[%t](%t, %layer, %rope, %ctx, %pool, %q, %k, %v, %qn, %kn, %info, %orig, %arena) : [index](index, index, i32, index, index, tensor<{t}x{QH * D}xf32>, tensor<{t}x{KVH * D}xf32>, tensor<{t}x{KVH * D}xf32>, tensor<128xf32>, tensor<128xf32>, tensor<{t}x4xi32>, tensor<64xi32>, tensor<1024xf16>)
  check.expect.close actual(%q) expected(%qe) atol(2.0e-3) rtol(2.0e-3) nan(different) : tensor<{t}x{QH * D}xf32>
  check.return
}}
""")
open(os.path.join(out, "cases.loom-test"), "w").write("\n".join(cases))
print("wrote", len(cases), "cases")
