#!/usr/bin/env python3
"""Roofline + target picker for a GGUF LLM on a Loom device.

usage: roofline.py MODEL.gguf [--cfg cfg.json] [--baseline perf.json] [--out roofline.json]
                   [--bw 216e9] [--flops 59e12] [--ctx 1000,2000,16000,32000]

Reads tensor sizes straight from the GGUF, plus (optionally) the port's cfg.json for
geometry the GGUF does not spell out (sliding_layers, head_dim_swa, kv_layers for KV
sharing). Prints, per metric, the hardware ceiling, the *practical* ceiling (ceiling x
the efficiency Loom has demonstrated on this hardware, see EFF), and - if --baseline
gives llama.cpp numbers - the target, stretch and margin over llama.cpp.

--baseline is a JSON object of llama.cpp ROCm numbers, keys as in perf.py output:
  {"decode_tps": 52.7, "decode_tps_2k": 48.7, "prefill_1000": 1593, "prefill_2000": 1678,
   "c4_steady": 157}
"""
import argparse, json, math, re, sys

# Demonstrated efficiencies on Strix Halo (fraction of the roofline actually reached by
# a tuned Loom port, end to end). Update these when a port beats them; cite the source.
EFF = {
    "decode": 0.90,      # dense. Bolt-E4B 61.1 / 68.1 (reference/calibration.md)
    "decode_moe": 0.78,  # expert gather. Flint 101 / 130
    "prefill": 0.33,     # end to end. Bolt-E4B with d256/d512 flash prefill: 2339 / 7094 (Flint 0.22-0.28)
    "c4": 0.70,          # dense. Bolt 0.65-0.66
    "c4_moe": 0.85,      # Flint 0.74, LFM2.5-24B-A2B 0.91
}
# Stretch: plausible with known-but-unbuilt levers (not yet demonstrated end to end).
STRETCH = {
    "decode": 0.95,      # small-op fusion + attention at the bandwidth limit
    "decode_moe": 0.88,  # batched expert gather
    "prefill": 0.45,     # GEMM efficiency already seen in isolation, plus flash attention
    "c4": 0.85,
    "c4_moe": 0.95,
}
GATHERED = re.compile(r"(^|\.)(token_embd|per_layer_token_embd)\.weight$")
EXPERT = re.compile(r"_exps\.weight$")


def load(path):
    import gguf
    r = gguf.GGUFReader(path)
    meta = {}
    for f in r.fields.values():
        try:
            v = f.contents()
        except Exception:
            continue
        meta[f.name] = v
    tensors = {t.name: (int(t.n_bytes), [int(x) for x in t.shape]) for t in r.tensors}
    return meta, tensors


def geometry(meta, cfg):
    arch = meta.get("general.architecture", "")
    g = lambda k, d=None: meta.get(f"{arch}.{k}", d)
    first = lambda v: v[0] if isinstance(v, list) else v
    L = int(cfg.get("layers", g("block_count")))
    hidden = int(cfg.get("hidden", g("embedding_length")))
    heads = int(cfg.get("heads", first(g("attention.head_count"))))
    kvh = int(cfg.get("kv_heads", first(g("attention.head_count_kv", heads))))
    hd = int(cfg.get("head_dim", g("attention.key_length", hidden // heads)))
    hd_swa = int(cfg.get("head_dim_swa", g("attention.key_length_swa", hd)))
    window = int(cfg.get("sliding_window", g("attention.sliding_window", 0)) or 0)
    pat = g("attention.sliding_window_pattern")
    if "sliding_layers" in cfg:
        sliding = set(cfg["sliding_layers"])
    elif isinstance(pat, list) and len(pat) == L:
        sliding = {i for i, s in enumerate(pat) if s}
    else:
        sliding = set()
    kv_layers = int(cfg.get("kv_layers", L))
    n_exp = int(g("expert_count", 0) or 0)
    n_used = int(g("expert_used_count", 0) or 0)
    return dict(arch=arch, L=L, hidden=hidden, heads=heads, kvh=kvh, hd=hd, hd_swa=hd_swa,
                window=window, sliding=sliding, kv_layers=kv_layers, n_exp=n_exp, n_used=n_used)


def weight_bytes(tensors, geo):
    """Bytes of weights streamed per decode step (1 token), and matmul params for flops."""
    tied = "output.weight" not in tensors
    read, params, head_bytes, detail = 0.0, 0.0, 0, {}
    for name, (nb, shape) in tensors.items():
        numel = math.prod(shape)
        if GATHERED.search(name):
            rows = shape[-1] if len(shape) > 1 else 1
            b = nb / rows                       # one row per token
            if tied and name == "token_embd.weight":
                b, head_bytes = nb, nb          # tied head reads it all
                params += numel
            kind = "embed/head"
        elif EXPERT.search(name) and geo["n_exp"]:
            b = nb * geo["n_used"] / geo["n_exp"]
            params += numel * geo["n_used"] / geo["n_exp"]
            kind = "experts (active)"
        else:
            b = nb
            if len(shape) > 1:
                params += numel
            if name == "output.weight":
                head_bytes = nb
                kind = "embed/head"
            else:
                kind = "attn" if ".attn_" in name else "ffn" if ".ffn_" in name else "other"
        read += b
        detail[kind] = detail.get(kind, 0) + b
    return read, params, head_bytes, detail


def attn_layers(tensors, L):
    """Layers that attend: those with a q projection. Hybrids (LFM2 short-conv, Mamba,
    delta-net) have attention in only some blocks; KV-shared layers still have q."""
    found = {i for i in range(L) if any(f"blk.{i}.attn_q{s}.weight" in tensors for s in ("", "kv"))}
    return found or set(range(L))


def kv_read_bytes(geo, ctx):
    """F16 K+V bytes read per decode token with `ctx` tokens cached (every attending layer)."""
    tot = 0
    for i in sorted(geo["attn"]):
        sl = i in geo["sliding"]
        hd = geo["hd_swa"] if sl else geo["hd"]
        n = min(ctx, geo["window"]) if (sl and geo["window"]) else ctx
        tot += 2 * geo["kvh"] * hd * 2 * n
    return tot


def attn_flops_per_token(geo, P):
    """Prefill attention flops per prompt token, causal, prompt length P."""
    tot = 0
    for i in sorted(geo["attn"]):
        sl = i in geo["sliding"]
        hd = geo["hd_swa"] if sl else geo["hd"]
        avg = min(P / 2, geo["window"]) if (sl and geo["window"]) else P / 2
        tot += 2 * 2 * geo["heads"] * hd * avg
    return tot


def pick_target(name, base, ceiling, eff, stretch):
    """target = ceiling x demonstrated efficiency; stretch = ceiling x STRETCH.

    The margin over llama.cpp is reported as is, with no cap and no label: it is
    whatever the hardware allows given llama.cpp's own efficiency (2x-6x is normal
    when llama.cpp is far from the roofline). The only class is 'at-ceiling'
    (< 3 % headroom: target parity and spend the effort elsewhere).
    """
    practical, st = ceiling * eff, ceiling * stretch
    r = dict(metric=name, ceiling=round(ceiling, 1), target=round(practical, 1),
             stretch=round(st, 1))
    if base is None:
        return r
    h = practical / base - 1
    r.update(baseline=base, baseline_eff=round(base / ceiling, 3),
             margin=round(h, 3), stretch_margin=round(st / base - 1, 3),
             at_ceiling=h < 0.03)
    if h < 0.03:
        r["target"] = base
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--cfg")
    ap.add_argument("--baseline")
    ap.add_argument("--out")
    ap.add_argument("--bw", type=float, default=216e9)
    ap.add_argument("--flops", type=float, default=59e12)
    ap.add_argument("--ctx", default="1000,2000,16000,32000")
    ap.add_argument("--budget-h", type=float, default=1.0,
                    help="the user's time budget for optimization (default 1 h); saved for progress.py")
    ap.add_argument("--goal", action="append", default=[],
                    help="user target, METRIC=TOK_S (e.g. decode_tps=70); replaces target/stretch for it")
    a = ap.parse_args()
    cfg = json.load(open(a.cfg)) if a.cfg else {}
    meta, tensors = load(a.gguf)
    geo = geometry(meta, cfg)
    geo["attn"] = attn_layers(tensors, geo["L"])
    wb, params, head, detail = weight_bytes(tensors, geo)
    base = json.load(open(a.baseline)) if a.baseline else {}
    ctxs = [int(x) for x in a.ctx.split(",")]

    rows = []
    dec = a.bw / wb
    key = "decode_moe" if geo["n_exp"] else "decode"
    ed, sd = EFF[key], STRETCH[key]
    rows.append(pick_target("decode_tps", base.get("decode_tps"), dec, ed, sd))
    for c in ctxs:
        k = f"decode_tps_{c // 1000}k"
        r = a.bw / (wb + kv_read_bytes(geo, c))
        rows.append(pick_target(k, base.get(k), r, ed, sd))
    for P in ctxs:
        k = f"prefill_{P}"
        hb = tensors.get("output.weight", tensors.get("token_embd.weight"))[1]
        f = 2 * (params - hb[0] * hb[1]) + attn_flops_per_token(geo, P)   # head: last token only
        r = a.flops / f
        rows.append(pick_target(k, base.get(k), r, EFF["prefill"], STRETCH["prefill"]))
    # 4 streams: one weight pass serves 4 tokens; KV at ~short context
    exp = detail.get("experts (active)", 0)
    if exp:   # 4 tokens route independently: ~min(n_exp, 4*k) distinct experts per layer
        n, k = geo["n_exp"], geo["n_used"]
        exp4 = exp * n * (1 - (1 - k / n) ** 4) / k            # expected distinct experts
    c4 = 4 * a.bw / (wb - exp + (exp4 if exp else 0) + 4 * kv_read_bytes(geo, 600))
    ck = "c4_moe" if geo["n_exp"] else "c4"
    rows.append(pick_target("c4_steady", base.get("c4_steady"), c4, EFF[ck], STRETCH[ck]))

    out = dict(gguf=a.gguf, geometry={k: (sorted(v) if isinstance(v, set) else v)
                                      for k, v in geo.items()},
               weight_bytes_per_token=round(wb), head_bytes=head,
               bytes_by_kind={k: round(v) for k, v in detail.items()},
               matmul_params=round(params), efficiency_assumed=EFF, stretch_assumed=STRETCH,
               bw=a.bw, flops=a.flops, metrics=rows,
               campaign=dict(budget_h=a.budget_h,
                             goals={g.split("=")[0]: float(g.split("=")[1]) for g in a.goal}))
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    print(f"weights read / decode token: {wb/1e9:.3f} GB  (head {head/1e9:.3f} GB)  "
          + "  ".join(f"{k} {v/1e9:.2f}" for k, v in detail.items()))
    print(f"matmul params: {params/1e9:.2f} B    BW {a.bw/1e9:.0f} GB/s   peak {a.flops/1e12:.0f} TFLOP/s")
    hdr = (f"{'metric':16} {'ceiling':>9} {'llama.cpp':>9} {'eff':>5} "
           f"{'TARGET':>9} {'margin':>8} {'stretch':>9} {'margin':>8}")
    print(hdr)
    for r in rows:
        b = r.get("baseline")
        line = f"{r['metric']:16} {r['ceiling']:9.1f} "
        if b:
            line += (f"{b:9.1f} {r['baseline_eff']:5.2f} {r['target']:9.1f} {r['margin']:+8.0%} "
                     f"{r['stretch']:9.1f} {r['stretch_margin']:+8.0%}"
                     + ("  at-ceiling: target parity" if r["at_ceiling"] else ""))
        else:
            line += f"{'-':>9} {'-':>5} {r['target']:9.1f} {'':>8} {r['stretch']:9.1f}"
        print(line)

if __name__ == "__main__":
    main()
