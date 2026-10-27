#!/usr/bin/env python3
"""Roofline + target picker for a GGUF LLM on a Loom device.

usage: roofline.py MODEL.gguf [--cfg cfg.json] [--baseline perf.json] [--out roofline.json]
                   [--bw 216e9] [--flops 59e12] [--ctx 1000,2000,16000,32000]

Reads tensor sizes straight from the GGUF, plus (optionally) the port's cfg.json for
geometry the GGUF does not spell out (sliding_layers, head_dim_swa, kv_layers for KV
sharing). Prints, per metric, the hardware ceiling, the *practical* ceiling (ceiling x
the efficiency Loom has demonstrated on this hardware, see EFF), and - if --baseline
gives llama.cpp numbers - a recommended target with a feasibility class.

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
    "prefill": 0.26,     # end to end; GEMMs alone reach ~0.5 of peak. Bolt 0.26, Flint 0.22-0.28
    "c4": 0.70,          # Bolt 0.65, Flint 0.74
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


def kv_read_bytes(geo, ctx):
    """F16 K+V bytes read per decode token with `ctx` tokens cached (every attending layer)."""
    tot = 0
    for i in range(geo["L"]):
        sl = i in geo["sliding"]
        hd = geo["hd_swa"] if sl else geo["hd"]
        n = min(ctx, geo["window"]) if (sl and geo["window"]) else ctx
        tot += 2 * geo["kvh"] * hd * 2 * n
    return tot


def attn_flops_per_token(geo, P):
    """Prefill attention flops per prompt token, causal, prompt length P."""
    tot = 0
    for i in range(geo["L"]):
        sl = i in geo["sliding"]
        hd = geo["hd_swa"] if sl else geo["hd"]
        avg = min(P / 2, geo["window"]) if (sl and geo["window"]) else P / 2
        tot += 2 * 2 * geo["heads"] * hd * avg
    return tot


def pick_target(name, base, practical, ceiling):
    """Target = practical ceiling (best efficiency Loom has demonstrated x roofline).

    Realistic: some port has already reached that efficiency on this hardware.
    Impressive: it is reported as a margin over llama.cpp, so classify the headroom
    (practical / baseline - 1) to decide where to spend effort:
      >= 25 %  'large'      headline metric; usually needs a new/retuned kernel
      10-25 %  'moderate'   tuning (dispatch shape, fusion) gets there
      3-10 %   'tight'      ablation-driven; budget-limited, may land short
      < 3 %    'at-ceiling' llama.cpp is near the roofline: target parity, spend effort elsewhere
    """
    if base is None:
        return dict(metric=name, ceiling=round(ceiling, 1), practical=round(practical, 1),
                    target=round(practical, 1), note="no baseline")
    h = practical / base - 1
    cls = ("large" if h >= 0.25 else "moderate" if h >= 0.10 else
           "tight" if h >= 0.03 else "at-ceiling")
    tgt = practical if h >= 0.03 else base
    return dict(metric=name, baseline=base, ceiling=round(ceiling, 1),
                practical=round(practical, 1), baseline_eff=round(base / ceiling, 3),
                headroom=round(h, 3), feasibility=cls, target=round(tgt, 1),
                target_vs_baseline=round(tgt / base - 1, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gguf")
    ap.add_argument("--cfg")
    ap.add_argument("--baseline")
    ap.add_argument("--out")
    ap.add_argument("--bw", type=float, default=216e9)
    ap.add_argument("--flops", type=float, default=59e12)
    ap.add_argument("--ctx", default="1000,2000,16000,32000")
    a = ap.parse_args()
    cfg = json.load(open(a.cfg)) if a.cfg else {}
    meta, tensors = load(a.gguf)
    geo = geometry(meta, cfg)
    wb, params, head, detail = weight_bytes(tensors, geo)
    base = json.load(open(a.baseline)) if a.baseline else {}
    ctxs = [int(x) for x in a.ctx.split(",")]

    rows = []
    dec = a.bw / wb
    ed = EFF["decode_moe"] if geo["n_exp"] else EFF["decode"]
    rows.append(pick_target("decode_tps", base.get("decode_tps"), dec * ed, dec))
    for c in ctxs:
        k = f"decode_tps_{c // 1000}k"
        r = a.bw / (wb + kv_read_bytes(geo, c))
        rows.append(pick_target(k, base.get(k), r * ed, r))
    for P in ctxs:
        k = f"prefill_{P}"
        hb = tensors.get("output.weight", tensors.get("token_embd.weight"))[1]
        f = 2 * (params - hb[0] * hb[1]) + attn_flops_per_token(geo, P)   # head: last token only
        r = a.flops / f
        rows.append(pick_target(k, base.get(k), r * EFF["prefill"], r))
    # 4 streams: one weight pass serves 4 tokens; KV at ~short context
    exp = detail.get("experts (active)", 0)
    if exp:   # 4 tokens route independently: ~min(n_exp, 4*k) distinct experts per layer
        exp4 = exp * min(geo["n_exp"], 4 * geo["n_used"]) / geo["n_used"]
    c4 = 4 * a.bw / (wb - exp + (exp4 if exp else 0) + 4 * kv_read_bytes(geo, 600))
    rows.append(pick_target("c4_steady", base.get("c4_steady"), c4 * EFF["c4"], c4))

    out = dict(gguf=a.gguf, geometry={k: (sorted(v) if isinstance(v, set) else v)
                                      for k, v in geo.items()},
               weight_bytes_per_token=round(wb), head_bytes=head,
               bytes_by_kind={k: round(v) for k, v in detail.items()},
               matmul_params=round(params), efficiency_assumed=EFF,
               bw=a.bw, flops=a.flops, metrics=rows)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
    print(f"weights read / decode token: {wb/1e9:.3f} GB  (head {head/1e9:.3f} GB)  "
          + "  ".join(f"{k} {v/1e9:.2f}" for k, v in detail.items()))
    print(f"matmul params: {params/1e9:.2f} B    BW {a.bw/1e9:.0f} GB/s   peak {a.flops/1e12:.0f} TFLOP/s")
    hdr = f"{'metric':16} {'ceiling':>9} {'practical':>9} {'llama.cpp':>9} {'eff':>6} {'headroom':>9} {'class':>11} {'TARGET':>9} {'vs base':>8}"
    print(hdr)
    for r in rows:
        b = r.get("baseline")
        print(f"{r['metric']:16} {r['ceiling']:9.1f} {r['practical']:9.1f} "
              + (f"{b:9.1f} {r['baseline_eff']:6.2f} {r['headroom']:+9.0%} {r['feasibility']:>11} "
                 f"{r['target']:9.1f} {r['target_vs_baseline']:+8.0%}" if b else
                 f"{'-':>9} {'-':>6} {'-':>9} {'-':>11} {r['target']:9.1f}"))


if __name__ == "__main__":
    main()
