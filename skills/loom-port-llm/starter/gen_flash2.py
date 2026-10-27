"""Generates flash2.loom: Glimmer prefill flash attention in transposed form.

Workgroup (16-token tile, KV head) of 16 waves; wave w owns query head
16*kv + w for the tile's 16 tokens. Each 32-key tile of K and V^T is staged
once in workgroup memory for all 16 heads. A wave computes S^T = K Q^T, so in
its MMA result each lane's column is one query (lane % 16) and its 8 elements
are keys 2i + lane/16 (the RDNA3 WMMA result layout). Softmax therefore needs
only in-lane reductions plus one xor-16 shuffle; P^T is assembled from the
lane pair and O^T += V^T P^T keeps the per-query rescale lane-uniform. Q stays
in registers (loaded straight from global memory), so no softmax scratch or Q
staging lives in workgroup memory. The output is scaled by sigmoid(gate).
Masks: causal, and on sliding layers keys > position - 2048.
"""
import json
import os
T = os.environ.get("FLASH2_TARGET", "llm")
if T == "llm":  # gen_llm.py: geometry from the LLM_CONFIG json (see inspect_gguf.py)
    _c = json.load(open(os.environ["LLM_CONFIG"]))
    CFG = dict(prefix="g", kvh=_c["kv_heads"], group=_c["heads"] // _c["kv_heads"], sinks=False,
               window=_c.get("sliding_window") or 262144, kv_f32=False, gate=_c.get("attn_gate", False), layers=_c["layers"] - 1)
else:
    CFG = {
        "glimmer": dict(prefix="g", kvh=2, group=16, sinks=False, window=2048, kv_f32=False, gate=True, layers=51),
        "flint": dict(prefix="flint", kvh=4, group=8, sinks=True, window=513, kv_f32=False, gate=False, layers=48),
    }[T]
P, KVH, G = CFG["prefix"], CFG["kvh"], CFG["group"]
QH = KVH * G
LANES = 32 * G
DPT = 128 * 32 // LANES          # staged dims per thread per key tile
TPK = 128 // DPT                 # threads per key
KT = int(os.environ.get("FLASH2_KT", "32"))   # keys per tile
KB = KT // 16
VTC, KBYTES, VTBYTES, KT_LESS = KT + 8, KT * 136 * 2, 128 * (KT + 8) * 2, KT - 1

U = []
A = U.append
A('''kernel.def export("{P}_attention_flash2") @{P}_attention_flash2(%tokens: index) {
  %c1 = index.constant 1 : index
  %ckv = index.constant {KVH} : index
  %c16 = index.constant 16 : index
  %clanes = index.constant {LANES} : index
  %tiles = index.div %tokens, %c16 : index
  kernel.launch.config workgroups(%tiles, %ckv, %c1) workgroup_size(%clanes, %c1, %c1) : index
} launch(%tokens_arg: index, %layer_arg: index, %sliding: i32, %context_arg: index, %pool_arg: index, %q: buffer, {"%sinks" if CFG["sinks"] else "%gate"}: buffer, %info: buffer, %origins: buffer, %arena: buffer, %out: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 16, 8192), multiple_of(%tokens_arg, 16)] : index
  %layer = index.assume %layer_arg [range(%layer_arg, 0, {CFG["layers"]})] : index
  %context = index.assume %context_arg [range(%context_arg, 16, 262144)] : index
  %pool = index.assume %pool_arg [range(%pool_arg, 16, 1048576)] : index
  %origin = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c8 = index.constant 8 : index
  %c15 = index.constant 15 : index
  %c16 = index.constant 16 : index
  %c31 = index.constant 31 : index
  %c32 = index.constant 32 : index
  %c128 = index.constant 128 : index
  %c0_i32 = scalar.constant 0 : i32
  %tile_count = index.div %tokens, %c16 : index
  %tile_id = kernel.workgroup.id<x> : index
  %tile = index.assume %tile_id [lt(%tile_id, %tile_count)] : index
  %kv_id = kernel.workgroup.id<y> : index
  %kv = index.assume %kv_id [range(%kv_id, 0, {KVH - 1})] : index
  %first_token = index.mul %tile, %c16 : index
  %uniform, %row32, %pos0_32 = func.call @{P}_tile_uniform(%info, %tokens, %first_token) : (buffer, index, index) -> (i1, i32, i32)
  scf.if %uniform {
  %workitem_id = kernel.workitem.id<x> : index
  %workitem = index.assume %workitem_id [range(%workitem_id, 0, {LANES - 1})] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, {G - 1})] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %lane_col = index.rem %lane, %c16 : index
  %lane_half = index.div %lane, %c16 : index
  %hi_half = index.cmp ne, %lane_half, %c0 : index
  %row_bound = scalar.assume %row32 [range(%row32, 0, 2147483646)] : i32
  %row = index.cast %row_bound : i32 to index
  %pos0_bound = scalar.assume %pos0_32 [range(%pos0_32, 0, 262128)] : i32
  %pos0 = index.cast %pos0_bound : i32 to index
  %pos_last = index.add %pos0, %c15 : index
  %padded = index.add %context, %c15 : index
  %blocks = index.div %padded, %c16 : index
  %k_plane = index.mul %layer, %c2 : index
  %v_plane = index.add %k_plane, %c1 : index
  %is_sliding = scalar.cmpi ne, %sliding, %c0_i32 : i32
  %window = index.constant {CFG["window"]} : index
  %window_less = index.constant {CFG["window"] - 1} : index
  %far = index.cmp uge, %pos0, %window : index
  %use_window = scalar.andi %is_sliding, %far : i1
  %window_start = index.sub %pos0, %window_less : index
  %key_low = scf.select %use_window, %window_start, %c0 : index
  %key_span = index.sub %pos_last, %key_low : index
  %key_count_value = index.add %key_span, %c1 : index
  %key_count = index.assume %key_count_value [range(%key_count_value, 1, 262144)] : index
  %ckt_less = index.constant {KT_LESS} : index
  %ckt = index.constant {KT} : index
  %key_tiles_round = index.add %key_count, %ckt_less : index
  %key_tiles = index.div %key_tiles_round, %ckt : index
  %c16h = index.constant {G} : index
  %head_value = index.madd %kv, %c16h, %wave : index
  %head = index.assume %head_value [range(%head_value, 0, {QH - 1})] : index

  // K [32 keys x 128 dims] and V^T [128 dims x 32 keys] tiles (F16, padded rows).
  %k_bytes = index.constant {KBYTES} : offset
  %k_tile = buffer.alloca<workgroup> align(16) %k_bytes : buffer
  %k_view = buffer.view %k_tile[%origin] : buffer -> view<{KT}x136xf16>
  %vt_bytes = index.constant {VTBYTES} : offset
  %vt_tile = buffer.alloca<workgroup> align(16) %vt_bytes : buffer
  %vt_view = buffer.view %vt_tile[%origin] : buffer -> view<128x{VTC}xf16>

  // Q^T as MMA rhs: lane column = token lane%16, 16 dims per fragment.
  %q_view = buffer.view %q[%origin] : buffer -> view<[%tokens]x{QH}x8x16xf32>
  %q_token_value = index.add %first_token, %lane_col : index
  %q_token = index.assume %q_token_value [lt(%q_token_value, %tokens)] : index''')
for c in range(8):
    A(f'''  %qc{c} = index.constant {c} : index
  %qraw{c} = vector.load %q_view[%q_token, %head, %qc{c}, %c0] : view<[%tokens]x{QH}x8x16xf32> -> vector<16xf32>
  %qh{c} = vector.fptrunc %qraw{c} : vector<16xf32> to vector<16xf16>
  %qf{c} = vector.fragment<rhs> %qh{c} shape [%c16, %c16] : vector<16xf16>''')
A('''  %zero8 = vector.constant 0.0 : vector<8xf32>
  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>
  %zero_f = scalar.constant 0.0 : f32
  %neg_big = scalar.constant -3.0e38 : f32
  %scale = scalar.constant 0.08838834764831845 : f32
  %scale_v = vector.splat %scale : vector<8xf32>
  %neg_v = vector.splat %neg_big : vector<8xf32>
  %zero_v8 = vector.constant 0.0 : vector<8xf32>
  %exchange = scalar.constant 16 : i32
  %width = scalar.constant 32 : i32
  %one_i32 = scalar.constant 1 : i32
  %two_i32 = scalar.constant 2 : i32
  %c2047_i32 = scalar.constant {CFG["window"] - 1} : i32
  %true_const = scalar.constant true : i1
  %not_sliding = scalar.xori %is_sliding, %true_const : i1
  %ns_v = vector.splat %not_sliding : vector<8xi1>
  // This lane's query position and its keys' row offsets within a 16-key block.
  %query_pos = index.add %pos0, %lane_col : index
  %qp32 = index.cast %query_pos : index to i32
  %qp_v = vector.splat %qp32 : vector<8xi32>
  %low32 = scalar.subi %qp32, %c2047_i32 : i32
  %low_v = vector.splat %low32 : vector<8xi32>
  %half32 = index.cast %lane_half : index to i32
  %row_off = vector.iota %half32 step %two_i32 : vector<8xi32>

  // Staging: thread t loads key t/{TPK}, dims (t%{TPK})*{DPT}.
  %ctpk = index.constant {TPK} : index
  %cdpt = index.constant {DPT} : index
  %st_key_value = index.div %workitem, %ctpk : index
  %st_key = index.assume %st_key_value [range(%st_key_value, 0, 31)] : index
  %st_seg = index.rem %workitem, %ctpk : index
  %st_dim_value = index.mul %st_seg, %cdpt : index
  %st_dim = index.assume %st_dim_value [range(%st_dim_value, 0, {128 - DPT})] : index
  %kz = vector.constant 0.0 : vector<{DPT}xf16>
''')
acc = ", ".join(f"%a{d} = %init : vector<8xf32>" for d in range(8))
if CFG["kv_f32"]:
    KV_LOAD = f"""      %k_cache = buffer.view %arena[%k_offset] : buffer -> view<{KVH}x128xf32>
      %v_cache = buffer.view %arena[%v_offset] : buffer -> view<{KVH}x128xf32>
      %kf = vector.load %k_cache[%kv, %st_dim] : view<{KVH}x128xf32> -> vector<{DPT}xf32>
      %vf = vector.load %v_cache[%kv, %st_dim] : view<{KVH}x128xf32> -> vector<{DPT}xf32>
      %kh = vector.fptrunc %kf : vector<{DPT}xf32> to vector<{DPT}xf16>
      %vh = vector.fptrunc %vf : vector<{DPT}xf32> to vector<{DPT}xf16>"""
else:
    KV_LOAD = f"""      %k_cache = buffer.view %arena[%k_offset] : buffer -> view<{KVH}x128xf16>
      %v_cache = buffer.view %arena[%v_offset] : buffer -> view<{KVH}x128xf16>
      %kh = vector.load %k_cache[%kv, %st_dim] : view<{KVH}x128xf16> -> vector<{DPT}xf16>
      %vh = vector.load %v_cache[%kv, %st_dim] : view<{KVH}x128xf16> -> vector<{DPT}xf16>"""
M_INIT, L_INIT = ("%sink", "%one_init") if CFG["sinks"] else ("%neg_big", "%zero_f")
if CFG["sinks"]:
    A('''  %sinks_view = buffer.view %sinks[%origin] : buffer -> view<32xf32>
  %sink = view.load %sinks_view[%head] : view<32xf32> -> f32
  %one_init = scalar.constant 1.0 : f32''')
LMAX = "\n".join(["    %lmax_c0 = scalar.addf %lmax0, %zero_f : f32"] + [f"    %lmax_c{kb} = scalar.maxnumf %lmax_c{kb - 1}, %lmax{kb} : f32" for kb in range(1, KB)] + [f"    %lmax = scalar.addf %lmax_c{KB - 1}, %zero_f : f32"])
LSUM = "\n".join(["    %lsum_c0 = scalar.addf %ls0, %zero_f : f32"] + [f"    %lsum_c{kb} = scalar.addf %lsum_c{kb - 1}, %ls{kb} : f32" for kb in range(1, KB)] + [f"    %lsum = scalar.addf %lsum_c{KB - 1}, %zero_f : f32"])
STAGE = f"""    %key_pos = index.add %tile_first, %st_key : index
    %key_valid = index.cmp ule, %key_pos, %pos_last : index
    %kvals, %vvals = scf.if %key_valid -> (vector<{DPT}xf16>, vector<{DPT}xf16>) {{
      %kp = index.assume %key_pos [range(%key_pos, 0, 262143)] : index
      %k_offset = func.call @{P}_kv_offset(%origins, %row, %kp, %blocks, %k_plane, %pool) : (buffer, index, index, index, index, index) -> (offset)
      %v_offset = func.call @{P}_kv_offset(%origins, %row, %kp, %blocks, %v_plane, %pool) : (buffer, index, index, index, index, index) -> (offset)
{KV_LOAD}
      scf.yield %kh, %vh : vector<{DPT}xf16>, vector<{DPT}xf16>
    }} else {{
      scf.yield %kz, %kz : vector<{DPT}xf16>, vector<{DPT}xf16>
    }}
    vector.store %kvals, %k_view[%st_key, %st_dim] : vector<{DPT}xf16>, view<{KT}x136xf16>
    scf.for %j = [%c0 to %cdpt step %c1] unroll {{
      %vj = vector.extract %vvals[%j] : vector<{DPT}xf16> -> f16
      %vt_row_value = index.add %st_dim, %j : index
      %vt_row = index.assume %vt_row_value [range(%vt_row_value, 0, 127)] : index
      view.store %vj, %vt_view[%vt_row, %st_key] : f16, view<128x{VTC}xf16>
    }}
"""
A(f'''  %m_final, %l_final, {", ".join(f"%o{d}" for d in range(8))} = scf.for %kt = [%c0 to %key_tiles step %c1](%m = {M_INIT} : f32, %l = {L_INIT} : f32, {acc}) -> (f32, f32, {", ".join(["vector<8xf32>"] * 8)}) {{
    %tile_first = index.madd %kt, %ckt, %key_low : index
''')
for sp in range(KT // 32):
    A(f'''    %spo{sp} = index.constant {sp * 32} : index
    %st_key{sp} = index.add %st_key, %spo{sp} : index
''' + STAGE.replace("%st_key", f"%st_key{sp}").replace("%key_pos", f"%key_pos{sp}").replace("%key_valid", f"%key_valid{sp}").replace("%kvals", f"%kvals{sp}").replace("%vvals", f"%vvals{sp}").replace("%kp ", f"%kp{sp} ").replace("%kp,", f"%kp{sp},").replace("%k_offset", f"%k_offset{sp}").replace("%v_offset", f"%v_offset{sp}").replace("%k_cache", f"%k_cache{sp}").replace("%v_cache", f"%v_cache{sp}").replace("%kh", f"%kh{sp}").replace("%vh", f"%vh{sp}").replace("%kf", f"%kf{sp}").replace("%vf", f"%vf{sp}").replace("%vj", f"%vj{sp}").replace("%vt_row", f"%vt_row{sp}").replace("%j ", f"%j{sp} ").replace("%j]", f"%j{sp}]").replace("%j :", f"%j{sp} :"))
A(f'''    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    %tf32 = index.cast %tile_first : index to i32''')
for kb in range(KB):
    acc_name = "%init"
    for c in range(8):
        A(f'''    %kl{kb}_{c} = vector.fragment.load<lhs> %k_view[%kb{kb}_row, %d{c}] shape [%c16, %c16] : view<{KT}x136xf16> -> vector<16xf16>
    %s{kb}_{c} = vector.mma %kl{kb}_{c}, %qf{c}, {acc_name} : vector<16xf16>, vector<16xf16>, vector<8xf32>''')
        acc_name = f"%s{kb}_{c}"
    A(f'''    %sc{kb} = vector.mulf %s{kb}_7, %scale_v : vector<8xf32>
    %base{kb} = scalar.addi %tf32, %kbo{kb} : i32
    %base{kb}_v = vector.splat %base{kb} : vector<8xi32>
    %kpos{kb} = vector.addi %base{kb}_v, %row_off : vector<8xi32>
    %causal{kb} = vector.cmpi sle, %kpos{kb}, %qp_v : vector<8xi32> -> vector<8xi1>
    %inwin{kb} = vector.cmpi sge, %kpos{kb}, %low_v : vector<8xi32> -> vector<8xi1>
    %winok{kb} = vector.ori %inwin{kb}, %ns_v : vector<8xi1>
    %valid{kb} = vector.andi %causal{kb}, %winok{kb} : vector<8xi1>
    %masked{kb} = vector.select %valid{kb}, %sc{kb}, %neg_v : vector<8xf32>
    %lmax{kb} = vector.reduce<maxnumf> %masked{kb}, %neg_big : vector<8xf32>, f32''')
A(LMAX + '''
    %omax, %omax_u = kernel.subgroup.shuffle<xor> %lmax, %exchange, %width : f32, i32, i32
    %tmax = scalar.maxnumf %lmax, %omax : f32
    %m_new = scalar.maxnumf %m, %tmax : f32
    %m_shift = scalar.subf %m, %m_new : f32
    %alpha = scalar.expf<afn> %m_shift : f32
    %m_new_v = vector.splat %m_new : vector<8xf32>''')
for kb in range(KB):
    A(f'''    %sh{kb} = vector.subf %masked{kb}, %m_new_v : vector<8xf32>
    %pe{kb} = vector.expf<afn> %sh{kb} : vector<8xf32>
    %p{kb} = vector.select %valid{kb}, %pe{kb}, %zero_v8 : vector<8xf32>
    %ls{kb} = vector.reduce<addf> %p{kb}, %zero_f : vector<8xf32>, f32''')
    # partner's 8 values (other key parity) via xor-16 shuffles
    for i in range(8):
        A(f'''    %pi{kb}_{i} = vector.extract %p{kb}[%d{i}i] : vector<8xf32> -> f32
    %po{kb}_{i}, %pu{kb}_{i} = kernel.subgroup.shuffle<xor> %pi{kb}_{i}, %exchange, %width : f32, i32, i32''')
    A(f'''    %partner{kb} = vector.from_elements {", ".join(f"%po{kb}_{i}" for i in range(8))} : vector<8xf32>
    %even{kb} = scf.select %hi_half, %partner{kb}, %p{kb} : vector<8xf32>
    %odd{kb} = scf.select %hi_half, %p{kb}, %partner{kb} : vector<8xf32>
    %pt{kb}_f = vector.interleave<0> %even{kb}, %odd{kb} : vector<8xf32>, vector<8xf32> -> vector<16xf32>
    %pt{kb}_h = vector.fptrunc %pt{kb}_f : vector<16xf32> to vector<16xf16>
    %pt{kb} = vector.fragment<rhs> %pt{kb}_h shape [%c16, %c16] : vector<16xf16>''')
A(LSUM + '''
    %osum, %osum_u = kernel.subgroup.shuffle<xor> %lsum, %exchange, %width : f32, i32, i32
    %tsum = scalar.addf %lsum, %osum : f32
    %l_scaled = scalar.mulf %l, %alpha : f32
    %l_new = scalar.addf %l_scaled, %tsum : f32
    %alpha_v = vector.splat %alpha : vector<8xf32>''')
for d in range(8):
    A(f"    %os{d} = vector.mulf %a{d}, %alpha_v : vector<8xf32>")
    prev = f"%os{d}"
    for kb in range(KB):
        name = f"%on{d}" if kb == KB - 1 else f"%on{d}_{kb}"
        A(f"""    %vl{d}_{kb} = vector.fragment.load<lhs> %vt_view[%d{d * 16 if d else 0}r, %kb{kb}_row] shape [%c16, %c16] : view<128x{{VTC}}xf16> -> vector<16xf16>
    {name} = vector.mma %vl{d}_{kb}, %pt{kb}, {prev} : vector<16xf16>, vector<16xf16>, vector<8xf32>""")
        prev = name
A('''    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    scf.yield %m_new, %l_new, ''' + ", ".join(f"%on{d}" for d in range(8)) + " : f32, f32, " + ", ".join(["vector<8xf32>"] * 8) + '''
  }
  // O^T / l (lane-uniform), gate, and store through a transposed strided view:
  // element (dim row, token column) -> out[first_token + col, head*128 + row].
  %one_f = scalar.constant 1.0 : f32
  %inv_l = scalar.divf %one_f, %l_final : f32
  %inv_v = vector.splat %inv_l : vector<8xf32>
  %c4096 = index.constant {QH * 128} : index
  %out_layout = encoding.layout.strided [%c1, %c4096] : encoding<layout>
  %out_first_a = index.mul %first_token, %c4096 : index
  %head_col = index.mul %head, %c128 : index
  %out_first = index.add %out_first_a, %head_col : index
  %four = index.constant 4 : offset
  %out_offset = index.scale %out_first, %four : index, offset -> offset
  %out_view = buffer.view %out[%out_offset] : buffer -> view<128x16xf32, %out_layout>''' + ("\n  %gate_view = buffer.view %gate[%out_offset] : buffer -> view<128x16xf32, %out_layout>" if CFG["gate"] else ""))
for d in range(8):
    r = f"%d{d * 16 if d else 0}r"
    gate_txt = f"""
  %gf{d} = vector.fragment.load<result> %gate_view[{r}, %c0] shape [%c16, %c16] : view<128x16xf32, %out_layout> -> vector<8xf32>
  %gs{d} = vector.logisticf<afn> %gf{d} : vector<8xf32>
  %of{d} = vector.mulf %ou{d}, %gs{d} : vector<8xf32>""" if CFG["gate"] else f"""
  %of{d} = vector.addf %ou{d}, %zero_v8 : vector<8xf32>"""
    A(f'''  %ou{d} = vector.mulf %o{d}, %inv_v : vector<8xf32>{gate_txt}
  vector.fragment.store<result> %of{d}, %out_view[{r}, %c0] shape [%c16, %c16] : vector<8xf32>, view<128x16xf32, %out_layout>''')
A("  }\n  kernel.return\n}\n")
text = "\n".join(U)
# constants used above
consts = "".join(f"  %d{c} = index.constant {c * 16} : index\n" for c in range(8))
consts += "".join(f"  %d{d * 16 if d else 0}r = index.constant {d * 16} : index\n" for d in range(8))
consts += "".join(f"  %d{i}i = index.constant {i} : index\n" for i in range(8))
consts += "".join(f"  %kb{kb}_row = index.constant {kb * 16} : index\n  %kbo{kb} = scalar.constant {kb * 16} : i32\n" for kb in range(KB))
text = text.replace("  %tile_count = index.div %tokens, %c16 : index\n", consts + "  %tile_count = index.div %tokens, %c16 : index\n", 1)
for key, value in {"{KT_LESS}": KT - 1, "{KT}": KT, "{KBYTES}": KT * 136 * 2, "{VTBYTES}": 128 * (KT + 8) * 2, "{VTC}": KT + 8, "{P}": P, "{KVH}": KVH, "{LANES}": LANES, "{G}": G, "{TPK}": TPK, "{DPT}": DPT,
                   '{CFG["layers"]}': CFG["layers"], '{CFG["window"]}': CFG["window"], '{CFG["window"] - 1}': CFG["window"] - 1,
                   "{G - 1}": G - 1, "{QH}": QH, "{QH - 1}": QH - 1, "{QH * 128}": QH * 128, "{KVH - 1}": KVH - 1, "{LANES - 1}": LANES - 1, "{128 - DPT}": 128 - DPT,
                   '{"%sinks" if CFG["sinks"] else "%gate"}': "%sinks" if CFG["sinks"] else "%gate"}.items():
    text = text.replace(key, str(value))
assert "{" not in "".join(l for l in text.split("\n") if "{P}" in l or "{KVH" in l)
out_path = os.environ.get("FLASH2_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "flash2.loom"))
open(out_path, "w", newline="\n").write(text)
