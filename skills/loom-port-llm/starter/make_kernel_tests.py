"""Kernel unit tests for a gen_llm.py build: numpy references from gguf dequantization.

usage: python make_kernel_tests.py CONFIG.json OUTDIR      (tools venv: gguf, numpy)
       run_kernel_tests.sh PACKAGE OUTDIR                  (loom-test on the GPU)
Covers g_embed, the decode matvecs (1- and 4-token variants) for one tensor of
each (format, K), the QK-norm + RoPE kernel (incl. head_dim 256 / partial
RoPE), and for hybrid models the whole gated delta-net chain on a span that
continues a stored recurrent state (conv, history commit, q/k norms,
recurrence, gated norm, published state). Extend it
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
VD = cfg["gdn"]["inner"] if cfg.get("gdn") else 0
for li in range(cfg["layers"]):
    for nm_, K in [("attn_q", H), ("attn_k", H), ("attn_v", H), ("ffn_down", cfg["ffn"]), ("attn_output", QH * D), ("ffn_gate", H),
                   ("attn_qkv", H), ("attn_gate", H), ("ssm_alpha", H), ("ssm_out", VD)]:
        name = f"blk.{li}.{nm_}.weight"
        if name not in T:
            continue
        key = (T[name].tensor_type.name.lower(), K)
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
A0 = next(i for i in range(cfg["layers"]) if i not in set(cfg.get("recurrent_layers", [])))
RD = cfg.get("rope_dims") or D
qn = deq(f"blk.{A0}.attn_q_norm.weight") if cfg["qk_norm"] else np.ones(D, np.float32)
kn = deq(f"blk.{A0}.attn_k_norm.weight") if cfg["qk_norm"] else np.ones(D, np.float32)
q = rng.standard_normal((t, QH, D)).astype(np.float32); k = rng.standard_normal((t, KVH, D)).astype(np.float32); v = rng.standard_normal((t, KVH, D)).astype(np.float32)
def norm(a, w): return a / np.sqrt((a * a).mean(-1, keepdims=True) + eps) * w
def rope(a):
    inv = cfg["rope_theta"] ** (-np.arange(0, RD // 2) * 2.0 / RD); ang = pos[:, None] * inv[None]
    c, s = np.cos(ang)[:, None], np.sin(ang)[:, None]
    if cfg["rope_type"] == "neox":
        a1, a2 = a[..., :RD // 2], a[..., RD // 2:RD]; return np.concatenate([a1 * c - a2 * s, a2 * c + a1 * s, a[..., RD:]], -1)
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
  %qn = check.file.read.npy path("r_qn.npy") : tensor<{D}xf32>
  %kn = check.file.read.npy path("r_kn.npy") : tensor<{D}xf32>
  %info = check.file.read.npy path("r_info.npy") : tensor<{t}x4xi32>
  %orig = check.file.read.npy path("r_orig.npy") : tensor<64xi32>
  %qe = check.file.read.npy path("r_qe.npy") : tensor<{t}x{QH * D}xf32>
  %arena = check.generate.fill value(0.0) : tensor<1024xf16>
  kernel.launch @g_qk_rope_store[%t](%t, %layer, %rope, %ctx, %pool, %q, %k, %v, %qn, %kn, %info, %orig, %arena) : [index](index, index, i32, index, index, tensor<{t}x{QH * D}xf32>, tensor<{t}x{KVH * D}xf32>, tensor<{t}x{KVH * D}xf32>, tensor<{D}xf32>, tensor<{D}xf32>, tensor<{t}x4xi32>, tensor<64xi32>, tensor<1024xf16>)
  check.expect.close actual(%q) expected(%qe) atol(2.0e-3) rtol(2.0e-3) nan(different) : tensor<{t}x{QH * D}xf32>
  check.return
}}
""")
# 3a. output head (argmax over the vocabulary) for one selected span of 3 tokens
t = 3
head_name = "output.weight" if "output.weight" in T else "token_embd.weight"
Wh = deq(head_name).astype(np.float64)
hs = rng.standard_normal((t, H)).astype(np.float32)
logits = Wh @ hs[-1].astype(np.float64)
meta = np.zeros(4 + 9, np.int32); meta[0] = 1; meta[4 + 0] = t; meta[4 + 6] = 1
order = np.argsort(-logits)
for n, a in [("hd_meta", meta), ("hd_h", hs), ("hd_w", raw(head_name)), ("hd_out", np.array([order[0]], np.int32))]:
    save(n, a)
CH = (cfg["vocab"] + 127) // 128
eos = set([cfg["eos_id"]] + cfg.get("extra_eos_ids", []))
print("head test: expected token", order[0], "logit gap to runner-up", float(logits[order[0]] - logits[order[1]]))
hwt = ty(raw(head_name))
cases.append(f"""check.case public @head_argmax {{
  %spans = check.literal value(1) : index
  %t = check.literal value({t}) : index
  %meta = check.file.read.npy path("hd_meta.npy") : tensor<13xi32>
  %h = check.file.read.npy path("hd_h.npy") : tensor<{t}x{H}xf32>
  %w = check.file.read.npy path("hd_w.npy") : {hwt}
  %expect = check.file.read.npy path("hd_out.npy") : tensor<1xi32>
  %partials = check.generate.fill value(0) : tensor<{CH * 2}xi32>
  %out = check.generate.fill value(0) : tensor<1xi32>
  kernel.launch @g_head_mv[%spans](%spans, %t, %meta, %h, %w, %partials) : [index](index, index, tensor<13xi32>, tensor<{t}x{H}xf32>, {hwt}, tensor<{CH * 2}xi32>)
  kernel.launch @g_head_select[%spans](%spans, %meta, %partials, %out) : [index](index, tensor<13xi32>, tensor<{CH * 2}xi32>, tensor<1xi32>)
  check.expect.equal actual(%out) expected(%expect) : tensor<1xi32>
  check.return
}}
""")
# 3b. attention chain: [q/gate split] -> qk norm + rope + paged KV store -> split-K attention -> combine (gate)
t, ctx = 4, 64
pos = np.arange(t)
gate_kind = cfg.get("attn_gate", False)
q = rng.standard_normal((t, QH, D)).astype(np.float32); k = rng.standard_normal((t, KVH, D)).astype(np.float32)
v = rng.standard_normal((t, KVH, D)).astype(np.float32); gate = rng.standard_normal((t, QH, D)).astype(np.float32)
qr = rope(norm(q, qn) if cfg["qk_norm"] else q); kr = rope(norm(k, kn) if cfg["qk_norm"] else k)
att = np.zeros((t, QH, D))
for hh in range(QH):
    kv = hh // (QH // KVH); sc = qr[:, hh] @ kr[:, kv].T / np.sqrt(D); sc[np.triu_indices(t, 1)] = -1e30
    pr = np.exp(sc - sc.max(-1, keepdims=True)); pr /= pr.sum(-1, keepdims=True); att[:, hh] = pr @ v[:, kv]
if gate_kind:
    att = att / (1 + np.exp(-gate))
blocks = (ctx + 15) // 16
orig = np.zeros(10 + blocks, np.int32); orig[8] = 40; orig[10:] = np.arange(blocks)  # record: map origin 40; identity page map
info = np.zeros((t, 4), np.int32); info[:, 1] = pos; info[:, 2] = 1
S = 64
for n, a in [("a_q", q.reshape(t, -1)), ("a_k", k.reshape(t, -1)), ("a_v", v.reshape(t, -1)), ("a_g", gate.reshape(t, -1)),
             ("a_qg", np.stack([q, gate], 2).reshape(t, -1)), ("a_info", info), ("a_orig", orig), ("a_o", att.reshape(t, -1).astype(np.float32))]:
    save(n, a)
QD = QH * D
split_text = ""
if gate_kind == "fused_q":
    split_text = f"""  %qg = check.file.read.npy path("a_qg.npy") : tensor<{t}x{2 * QD}xf32>
  kernel.launch @g_split_qgate[%t](%t, %qg, %q, %g) : [index](index, tensor<{t}x{2 * QD}xf32>, tensor<{t}x{QD}xf32>, tensor<{t}x{QD}xf32>)
"""
cases.append(f"""check.case public @attention_chain {{
  %t = check.literal value({t}) : index
  %layer = check.literal value(0) : index
  %rope = check.literal value(1) : i32
  %sliding = check.literal value(0) : i32
  %ctx = check.literal value({ctx}) : index
  %pool = check.literal value({ctx}) : index
  %q = {"check.generate.fill value(0.0)" if gate_kind == "fused_q" else 'check.file.read.npy path("a_q.npy")'} : tensor<{t}x{QD}xf32>
  %g = {"check.generate.fill value(0.0)" if gate_kind == "fused_q" else 'check.file.read.npy path("a_g.npy")'} : tensor<{t}x{QD}xf32>
  %k = check.file.read.npy path("a_k.npy") : tensor<{t}x{KVH * D}xf32>
  %v = check.file.read.npy path("a_v.npy") : tensor<{t}x{KVH * D}xf32>
  %qn = check.file.read.npy path("r_qn.npy") : tensor<{D}xf32>
  %kn = check.file.read.npy path("r_kn.npy") : tensor<{D}xf32>
  %info = check.file.read.npy path("a_info.npy") : tensor<{t}x4xi32>
  %orig = check.file.read.npy path("a_orig.npy") : tensor<{10 + blocks}xi32>
  %eo = check.file.read.npy path("a_o.npy") : tensor<{t}x{QD}xf32>
  %arena = check.generate.fill value(0.0) : tensor<{2 * ctx * KVH * D}xf16>
  %pacc = check.generate.fill value(0.0) : tensor<{t * QH * S * D}xf32>
  %pml = check.generate.fill value(0.0) : tensor<{t * QH * S * 2}xf32>
  %o = check.generate.fill value(0.0) : tensor<{t}x{QD}xf32>
{split_text}  kernel.launch @g_qk_rope_store[%t](%t, %layer, %rope, %ctx, %pool, %q, %k, %v, %qn, %kn, %info, %orig, %arena) : [index](index, index, i32, index, index, tensor<{t}x{QD}xf32>, tensor<{t}x{KVH * D}xf32>, tensor<{t}x{KVH * D}xf32>, tensor<{D}xf32>, tensor<{D}xf32>, tensor<{t}x4xi32>, tensor<{10 + blocks}xi32>, tensor<{2 * ctx * KVH * D}xf16>)
  kernel.launch @g_attention_split[%t](%t, %layer, %sliding, %ctx, %pool, %q, %info, %orig, %arena, %pacc, %pml) : [index](index, index, i32, index, index, tensor<{t}x{QD}xf32>, tensor<{t}x4xi32>, tensor<{10 + blocks}xi32>, tensor<{2 * ctx * KVH * D}xf16>, tensor<{t * QH * S * D}xf32>, tensor<{t * QH * S * 2}xf32>)
  kernel.launch @g_attention_combine[%t](%t, %sliding, %info, %pacc, %pml, {"%g" if gate_kind else "%q"}, %o) : [index](index, i32, tensor<{t}x4xi32>, tensor<{t * QH * S * D}xf32>, tensor<{t * QH * S * 2}xf32>, tensor<{t}x{QD}xf32>, tensor<{t}x{QD}xf32>)
  check.expect.close actual(%o) expected(%eo) atol(3.0e-3) rtol(3.0e-3) nan(different) : tensor<{t}x{QD}xf32>
  check.return
}}
""")
# 4. gated delta-net chain on one 5-token span at position 3 that continues the reader slot
if cfg.get("recurrent_layers"):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gen_gdn import gdn_geometry
    G = gdn_geometry(cfg); HK, HV, DK, DV, K, C, VD = (G[k] for k in ("HK", "HV", "DK", "DV", "K", "C", "VD"))
    li = cfg["recurrent_layers"][0]; b = f"blk.{li}."
    conv_w = deq(b + "ssm_conv1d.weight").reshape(C, K).astype(np.float32)
    a_log = deq(b + "ssm_a").astype(np.float32); dt = deq(b + "ssm_dt.bias").astype(np.float32); nw = deq(b + "ssm_norm.weight").astype(np.float32)
    t, p0 = 5, 3
    qkv = rng.standard_normal((t, C)).astype(np.float32); hist0 = rng.standard_normal((K - 1, C)).astype(np.float32)
    S0 = (0.1 * rng.standard_normal((HV, DV, DK))).astype(np.float32)  # [head][b][a]
    alpha = rng.standard_normal((t, HV)).astype(np.float32); beta = rng.standard_normal((t, HV)).astype(np.float32)
    z = rng.standard_normal((t, VD)).astype(np.float32)
    full = np.concatenate([hist0, qkv], 0)
    conv = np.stack([(full[i:i + K] * conv_w.T).sum(0) for i in range(t)])
    silu = conv / (1 + np.exp(-conv)); hist1 = full[-(K - 1):]
    qs, ks, vs = silu[:, :HK * DK].reshape(t, HK, DK), silu[:, HK * DK:2 * HK * DK].reshape(t, HK, DK), silu[:, 2 * HK * DK:].reshape(t, HV, DV)
    qinv = 1 / np.sqrt((qs ** 2).sum(-1) + eps); kinv = 1 / np.sqrt((ks ** 2).sum(-1) + eps)
    inv = np.stack([qinv, kinv], -1).reshape(t, 2 * HK)
    S = S0.astype(np.float64).copy(); raw_o = np.zeros((t, HV, DV))
    for i in range(t):
        for h in range(HV):
            kh = h % HK; q = qs[i, kh] * qinv[i, kh] / np.sqrt(DK); k = ks[i, kh] * kinv[i, kh]
            g = np.log1p(np.exp(alpha[i, h] + dt[h])) * a_log[h]
            St = S[h] * np.exp(g)                      # [b][a]
            d = (vs[i, h] - St @ k) / (1 + np.exp(-beta[i, h]))
            St = St + np.outer(d, k); S[h] = St; raw_o[i, h] = St @ q
    o = raw_o / np.sqrt((raw_o ** 2).mean(-1, keepdims=True) + eps) * nw * (z.reshape(t, HV, DV) / (1 + np.exp(-z.reshape(t, HV, DV))))
    LB, SB = G["layer_bytes"], G["state_bytes"]
    arena = np.zeros(2 * LB // 4, np.float32)
    arena[:SB // 4] = S0.reshape(-1); arena[SB // 4:SB // 4 + (K - 1) * C] = hist0.reshape(-1)
    expect = arena.copy(); expect[LB // 4:LB // 4 + SB // 4] = S.astype(np.float32).reshape(-1)
    expect[LB // 4 + SB // 4:LB // 4 + SB // 4 + (K - 1) * C] = hist1.reshape(-1)
    meta = np.zeros(4 + 9, np.int32); meta[0] = 1; meta[4 + 0] = t; meta[4 + 1] = p0; meta[4 + 3] = 0; meta[4 + 4] = 0
    origins = np.zeros(5, np.int64); origins[0] = 0; origins[1] = LB; origins[3] = p0
    info = np.zeros((t, 4), np.int32); info[:, 1] = p0 + np.arange(t); info[:, 2] = 1
    for n, a in [("g_meta", meta), ("g_orig", origins.view(np.int32)), ("g_info", info), ("g_arena", arena), ("g_expect", expect),
                 ("g_qkv", qkv), ("g_conv_w", conv_w.reshape(-1)), ("g_silu", silu.astype(np.float32)), ("g_inv", inv.astype(np.float32)),
                 ("g_alpha", alpha), ("g_beta", beta), ("g_a", a_log), ("g_dt", dt), ("g_nw", nw), ("g_z", z),
                 ("g_raw", raw_o.reshape(t, VD).astype(np.float32)), ("g_o", o.reshape(t, VD).astype(np.float32))]:
        save(n, a)
    AT = f"tensor<{2 * LB // 4}xf32>"
    cases.append(f"""check.case public @gdn_chain {{
  %t = check.literal value({t}) : index
  %spans = check.literal value(1) : index
  %layer = check.literal value(0) : index
  %meta = check.file.read.npy path("g_meta.npy") : tensor<13xi32>
  %orig = check.file.read.npy path("g_orig.npy") : tensor<10xi32>
  %info = check.file.read.npy path("g_info.npy") : tensor<{t}x4xi32>
  %arena = check.file.read.npy path("g_arena.npy") : {AT}
  %expect = check.file.read.npy path("g_expect.npy") : {AT}
  %qkv = check.file.read.npy path("g_qkv.npy") : tensor<{t}x{C}xf32>
  %w = check.file.read.npy path("g_conv_w.npy") : tensor<{C * K}xf32>
  %esilu = check.file.read.npy path("g_silu.npy") : tensor<{t}x{C}xf32>
  %einv = check.file.read.npy path("g_inv.npy") : tensor<{t}x{2 * HK}xf32>
  %alpha = check.file.read.npy path("g_alpha.npy") : tensor<{t}x{HV}xf32>
  %beta = check.file.read.npy path("g_beta.npy") : tensor<{t}x{HV}xf32>
  %a = check.file.read.npy path("g_a.npy") : tensor<{HV}xf32>
  %dt = check.file.read.npy path("g_dt.npy") : tensor<{HV}xf32>
  %nw = check.file.read.npy path("g_nw.npy") : tensor<{DV}xf32>
  %z = check.file.read.npy path("g_z.npy") : tensor<{t}x{VD}xf32>
  %eraw = check.file.read.npy path("g_raw.npy") : tensor<{t}x{VD}xf32>
  %eo = check.file.read.npy path("g_o.npy") : tensor<{t}x{VD}xf32>
  %silu = check.generate.fill value(0.0) : tensor<{t}x{C}xf32>
  %inv = check.generate.fill value(0.0) : tensor<{t}x{2 * HK}xf32>
  %raw = check.generate.fill value(0.0) : tensor<{t}x{VD}xf32>
  %o = check.generate.fill value(0.0) : tensor<{t}x{VD}xf32>
  kernel.launch @g_gdn_conv[%t](%t, %spans, %layer, %meta, %info, %orig, %arena, %qkv, %w, %silu) : [index](index, index, index, tensor<13xi32>, tensor<{t}x4xi32>, tensor<10xi32>, {AT}, tensor<{t}x{C}xf32>, tensor<{C * K}xf32>, tensor<{t}x{C}xf32>)
  kernel.launch @g_gdn_conv_commit[%spans](%t, %spans, %layer, %meta, %orig, %arena, %qkv) : [index](index, index, index, tensor<13xi32>, tensor<10xi32>, {AT}, tensor<{t}x{C}xf32>)
  kernel.launch @g_gdn_qk_norm[%t](%t, %silu, %inv) : [index](index, tensor<{t}x{C}xf32>, tensor<{t}x{2 * HK}xf32>)
  kernel.launch @g_gdn_recurrence[%spans](%t, %spans, %layer, %meta, %orig, %arena, %silu, %inv, %alpha, %beta, %a, %dt, %raw) : [index](index, index, index, tensor<13xi32>, tensor<10xi32>, {AT}, tensor<{t}x{C}xf32>, tensor<{t}x{2 * HK}xf32>, tensor<{t}x{HV}xf32>, tensor<{t}x{HV}xf32>, tensor<{HV}xf32>, tensor<{HV}xf32>, tensor<{t}x{VD}xf32>)
  kernel.launch @g_gdn_norm_gate[%t](%t, %info, %raw, %nw, %z, %o) : [index](index, tensor<{t}x4xi32>, tensor<{t}x{VD}xf32>, tensor<{DV}xf32>, tensor<{t}x{VD}xf32>, tensor<{t}x{VD}xf32>)
  check.expect.close actual(%silu) expected(%esilu) atol(1.0e-4) rtol(1.0e-3) nan(different) : tensor<{t}x{C}xf32>
  check.expect.close actual(%inv) expected(%einv) atol(1.0e-4) rtol(1.0e-3) nan(different) : tensor<{t}x{2 * HK}xf32>
  check.expect.close actual(%raw) expected(%eraw) atol(1.0e-4) rtol(2.0e-3) nan(different) : tensor<{t}x{VD}xf32>
  check.expect.close actual(%o) expected(%eo) atol(1.0e-3) rtol(2.0e-3) nan(different) : tensor<{t}x{VD}xf32>
  check.expect.close actual(%arena) expected(%expect) atol(1.0e-4) rtol(1.0e-3) nan(different) : {AT}
  check.return
}}
""")
import re
# hrx-loom 0.1.0: float comparison is check.expect.approximate with the actual and expected types.
text = re.sub(r"check\.expect\.close (actual\(.*\)) : (.*)$", r"check.expect.approximate \1 : \2, \2", "\n".join(cases), flags=re.M)
open(os.path.join(out, "cases.loom-test"), "w", newline="\n").write(text)
print("wrote", len(cases), "cases")
