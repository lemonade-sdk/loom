"""Builds flash.loom: the prefill flash-attention kernel (preamble in flash_pre.loom.txt)."""
import sys
pre = open(__file__.rsplit("/", 1)[0] + "/flash_pre.loom.txt").read()
L = []
L.append("    // S = Q K^T for keys [0, 16) and [16, 32): rhs lane reads key row, 16 dims.")
L.append("    %key_row0 = index.add %lane_row, %c0 : index")
L.append("    %key_row1 = index.add %lane_row, %c16 : index")
for half in range(2):
    acc = "%init"
    for c in range(8):
        L.append(f"    %kr{half}_{c} = vector.load %k_view[%key_row{half}, %d{c*16 if c else 0}] : view<32x136xf16> -> vector<16xf16>".replace("%d0]", "%c0]"))
        L.append(f"    %kb{half}_{c} = vector.fragment<rhs> %kr{half}_{c} shape [%c16, %c16] : vector<16xf16>")
        L.append(f"    %s{half}_{c} = vector.mma %qf{c}, %kb{half}_{c}, {acc} : vector<16xf16>, vector<16xf16>, vector<8xf32>")
        acc = f"%s{half}_{c}"
L.append("    vector.fragment.store<result> %s0_7, %s_view[%c0, %c0] shape [%c16, %c16] : vector<8xf32>, view<16x32xf32>")
L.append("    vector.fragment.store<result> %s1_7, %s_view[%c0, %c16] shape [%c16, %c16] : vector<8xf32>, view<16x32xf32>")
L.append("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
# Online softmax for this lane's row segment.
L.append('''    %scores = vector.load %s_view[%sm_row, %sm_col] : view<16x32xf32> -> vector<16xf32>
    %scales = vector.splat %scale : vector<16xf32>
    %scaled = vector.mulf %scores, %scales : vector<16xf32>
    %col0 = index.add %tile_first, %sm_col : index
    %col0_i32 = index.cast %col0 : index to i32
    %one_i32 = scalar.constant 1 : i32
    %key_positions = vector.iota %col0_i32 step %one_i32 : vector<16xi32>
    %qp32 = index.cast %query_pos : index to i32
    %qp_v = vector.splat %qp32 : vector<16xi32>
    %causal = vector.cmpi sle, %key_positions, %qp_v : vector<16xi32> -> vector<16xi1>
    %c512_i32 = scalar.constant 512 : i32
    %low32 = scalar.subi %qp32, %c512_i32 : i32
    %low_v = vector.splat %low32 : vector<16xi32>
    %in_window = vector.cmpi sge, %key_positions, %low_v : vector<16xi32> -> vector<16xi1>
    %not_sliding = scalar.xori %is_sliding, %true_const : i1
    %ns_v = vector.splat %not_sliding : vector<16xi1>
    %window_ok = vector.ori %in_window, %ns_v : vector<16xi1>
    %valid = vector.andi %causal, %window_ok : vector<16xi1>
    %neg_v = vector.splat %neg_big : vector<16xf32>
    %masked = vector.select %valid, %scaled, %neg_v : vector<16xf32>
    %local_max = vector.reduce<maxnumf> %masked, %neg_big : vector<16xf32>, f32
    %other_max, %unused_max = kernel.subgroup.shuffle<xor> %local_max, %exchange, %width : f32, i32, i32
    %tile_max = scalar.maxnumf %local_max, %other_max : f32
    %m_new = scalar.maxnumf %m, %tile_max : f32
    %m_shift = scalar.subf %m, %m_new : f32
    %alpha = scalar.expf<afn> %m_shift : f32
    %m_new_v = vector.splat %m_new : vector<16xf32>
    %shifted = vector.subf %masked, %m_new_v : vector<16xf32>
    %p = vector.expf<afn> %shifted : vector<16xf32>
    %zero16 = vector.constant 0.0 : vector<16xf32>
    %p_masked = vector.select %valid, %p, %zero16 : vector<16xf32>
    %local_sum = vector.reduce<addf> %p_masked, %zero_f : vector<16xf32>, f32
    %other_sum, %unused_sum = kernel.subgroup.shuffle<xor> %local_sum, %exchange, %width : f32, i32, i32
    %tile_sum = scalar.addf %local_sum, %other_sum : f32
    %l_scaled = scalar.mulf %l, %alpha : f32
    %l_new = scalar.addf %l_scaled, %tile_sum : f32
    %p_half = vector.fptrunc %p_masked : vector<16xf32> to vector<16xf16>
    vector.store %p_half, %p_view[%sm_row, %sm_col] : vector<16xf16>, view<16x40xf16>
    %alpha16 = vector.splat %alpha : vector<16xf32>
    %a_half_col = index.rem %sm_col, %c16 : index
    vector.store %alpha16, %a_view[%sm_row, %c0] : vector<16xf32>, view<16x16xf32>
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
    %alpha_frag = vector.fragment.load<result> %a_view[%c0, %c0] shape [%c16, %c16] : view<16x16xf32> -> vector<8xf32>
    %pf0 = vector.fragment.load<lhs> %p_view[%c0, %c0] shape [%c16, %c16] : view<16x40xf16> -> vector<16xf16>
    %pf1 = vector.fragment.load<lhs> %p_view[%c0, %c16] shape [%c16, %c16] : view<16x40xf16> -> vector<16xf16>''')
for d in range(8):
    L.append(f"    %os{d} = vector.mulf %a{d}, %alpha_frag : vector<8xf32>")
    L.append(f"    %vrow{d}_value = index.add %lane_row, %d{d*16 if d else 0} : index".replace("%d0 :", "%c0 :"))
    L.append(f"    %vrow{d} = index.assume %vrow{d}_value [range(%vrow{d}_value, 0, 127)] : index")
    L.append(f"    %vr{d}a = vector.load %vt_view[%vrow{d}, %c0] : view<128x40xf16> -> vector<16xf16>")
    L.append(f"    %vr{d}b = vector.load %vt_view[%vrow{d}, %c16] : view<128x40xf16> -> vector<16xf16>")
    L.append(f"    %vb{d}a = vector.fragment<rhs> %vr{d}a shape [%c16, %c16] : vector<16xf16>")
    L.append(f"    %vb{d}b = vector.fragment<rhs> %vr{d}b shape [%c16, %c16] : vector<16xf16>")
    L.append(f"    %on{d}a = vector.mma %pf0, %vb{d}a, %os{d} : vector<16xf16>, vector<16xf16>, vector<8xf32>")
    L.append(f"    %on{d} = vector.mma %pf1, %vb{d}b, %on{d}a : vector<16xf16>, vector<16xf16>, vector<8xf32>")
L.append("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
L.append("    scf.yield %m_new, %l_new, " + ", ".join(f"%on{d}" for d in range(8)) + " : f32, f32, " + ", ".join(["vector<8xf32>"] * 8))
L.append("  }")
L.append('''  // O /= l per row, then store this wave's 16 tokens x 128 dims of head `head`.
  %inv_l = scalar.divf %one_f, %l_final : f32
  %inv16 = vector.splat %inv_l : vector<16xf32>
  vector.store %inv16, %a_view[%sm_row, %c0] : vector<16xf32>, view<16x16xf32>
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %inv_frag = vector.fragment.load<result> %a_view[%c0, %c0] shape [%c16, %c16] : view<16x16xf32> -> vector<8xf32>
  %c4096 = index.constant 4096 : index
  %out_layout = encoding.layout.strided [%c4096, %c1] : encoding<layout>
  %out_first_a = index.mul %first_token, %c4096 : index
  %head_col = index.mul %head, %c128 : index
  %out_first = index.add %out_first_a, %head_col : index
  %four = index.constant 4 : offset
  %out_offset = index.scale %out_first, %four : index, offset -> offset
  %out_view = buffer.view %out[%out_offset] : buffer -> view<16x128xf32, %out_layout>''')
for d in range(8):
    col = f"%d{d*16}" if d else "%c0"
    L.append(f"  %of{d} = vector.mulf %o{d}, %inv_frag : vector<8xf32>")
    L.append(f"  vector.fragment.store<result> %of{d}, %out_view[%c0, {col}] shape [%c16, %c16] : vector<8xf32>, view<16x128xf32, %out_layout>")
L.append("  }\n  kernel.return\n}\n")
body = "\n".join(L)
pre2 = pre.replace("  %zero8 = vector.constant 0.0 : vector<8xf32>", "  %zero8 = vector.constant 0.0 : vector<8xf32>\n  %zero_f = scalar.constant 0.0 : f32\n  %true_const = scalar.constant true : i1")
sys.stdout.write(pre2 + body)
