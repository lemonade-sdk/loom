"""Generates Glimmer prefill GEMM kernels (adapted from the Flint port) (WMMA, RDNA3 wave32) as Loom source.

Y[rows, n] (+)= X[rows, k] . W^T (X: F16, pre-scaled per row by g_row_scale_f16) with GGUF Q8_0 or Q4_K weights dequantized to
F16 in workgroup memory. A 256-thread workgroup owns a 64x64 output tile;
eight waves own a 4x2 grid of 16x32 outputs. K advances 64 values per stage.
Rows and outputs must be multiples of 64; k a multiple of 64 (Q8_0) or 256 (Q4_K).

Variants:
  store / acc   single weight, Y = XW^T or Y += XW^T
  swiglu        two weights (gate, up), Y = silu(X Wg^T) * (X Wu^T)
"""
import os, sys
PREFIX = os.environ.get("GEMM_PREFIX", "g")
META = os.environ.get("GEMM_KMETA", "@g_k_metadata")

FMT = {
    "q4_1": dict(block_bytes=20, block_values=32),
    "q8_0": dict(block_bytes=34, block_values=32),
    "q4_k": dict(block_bytes=144, block_values=256),
    "q6_k": dict(block_bytes=210, block_values=256),
}

def dequant16(fmt, w, view_name, out):
    """Loom snippet: dequantize 16 consecutive weights of row %w_row at column %w_col into F16 vector %{out}."""
    if fmt == "q4_1":
        return f'''
      %{out}_block = index.div %w_col, %c32 : index
      %{out}_in_block = index.rem %w_col, %c32 : index
      %{out}_half = index.div %{out}_in_block, %c16 : index
      %{out}_block_add = index.scale %{out}_block, %block_bytes : index, offset -> offset
      %{out}_block_origin = index.add %w_row_origin, %{out}_block_add : offset
      %{out}_four = index.constant 4 : offset
      %{out}_code_origin = index.add %{out}_block_origin, %{out}_four : offset
      %{out}_dm_view = buffer.view %{w}[%{out}_block_origin] : buffer -> view<2xf16>
      %{out}_code_view = buffer.view %{w}[%{out}_code_origin] : buffer -> view<4xi32>
      %{out}_d = view.load %{out}_dm_view[0] : view<2xf16> -> f16
      %{out}_m = view.load %{out}_dm_view[1] : view<2xf16> -> f16
      %{out}_packed = vector.load %{out}_code_view[0] : view<4xi32> -> vector<4xi32>
      %{out}_shift_bits = index.mul %{out}_half, %c4 : index
      %{out}_shift_i32 = index.cast %{out}_shift_bits : index to i32
      %{out}_shift = vector.splat %{out}_shift_i32 : vector<4xi32>
      %{out}_shifted = vector.shrui %{out}_packed, %{out}_shift : vector<4xi32>
      %{out}_masked = vector.andi %{out}_shifted, %nibble_mask : vector<4xi32>
      %{out}_bytes = vector.bitcast %{out}_masked : vector<4xi32> to vector<16xi8>
      %{out}_codes = vector.uitofp %{out}_bytes : vector<16xi8> to vector<16xf16>
      %{out}_ds = vector.splat %{out}_d : vector<16xf16>
      %{out}_ms = vector.splat %{out}_m : vector<16xf16>
      %{out} = vector.fmaf %{out}_codes, %{out}_ds, %{out}_ms : vector<16xf16>'''
    if fmt == "q8_0":
        return f'''
      %{out}_block = index.div %w_col, %c32 : index
      %{out}_half = index.rem %w_col, %c32 : index
      %{out}_block_add = index.scale %{out}_block, %block_bytes : index, offset -> offset
      %{out}_block_origin = index.add %w_row_origin, %{out}_block_add : offset
      %{out}_half_bytes = index.cast %{out}_half : index to offset
      %{out}_code_origin0 = index.add %{out}_block_origin, %two : offset
      %{out}_code_origin = index.add %{out}_code_origin0, %{out}_half_bytes : offset
      %{out}_scale_view = buffer.view %{w}[%{out}_block_origin] : buffer -> view<1xf16>
      %{out}_code_view = buffer.view %{w}[%{out}_code_origin] : buffer -> view<16xi8>
      %{out}_scale_half = view.load %{out}_scale_view[0] : view<1xf16> -> f16
      %{out}_scale = scalar.extf %{out}_scale_half : f16 to f32
      %{out}_codes = vector.load %{out}_code_view[0] : view<16xi8> -> vector<16xi8>
      %{out}_values = vector.sitofp %{out}_codes : vector<16xi8> to vector<16xf32>
      %{out}_scales = vector.splat %{out}_scale : vector<16xf32>
      %{out}_scaled = vector.mulf %{out}_values, %{out}_scales : vector<16xf32>
      %{out} = vector.fptrunc %{out}_scaled : vector<16xf32> to vector<16xf16>'''
    if fmt == "q6_k":
        # Q6_K: value c of a superblock: half = c/128, j = (c%128)/32, l = c%32;
        # q = ((ql[half*64 + (j%2)*32 + l] >> 4*(j/2)) & 15) | (((qh[half*32 + l] >> 2j) & 3) << 4);
        # w = d * sc[half*8 + 2j + l/16] * (q - 32). 16 consecutive values share one scale.
        return f'''
      %{out}_c128 = index.constant 128 : index
      %{out}_c64 = index.constant 64 : index
      %{out}_qh_base = index.constant 128 : index
      %{out}_sc_base = index.constant 192 : index
      %{out}_d_bytes = index.constant 208 : offset
      %{out}_super = index.div %w_col, %c256 : index
      %{out}_within = index.rem %w_col, %c256 : index
      %{out}_half = index.div %{out}_within, %{out}_c128 : index
      %{out}_in_half = index.rem %{out}_within, %{out}_c128 : index
      %{out}_j = index.div %{out}_in_half, %c32 : index
      %{out}_l0 = index.rem %{out}_in_half, %c32 : index
      %{out}_jl = index.rem %{out}_j, %c2 : index
      %{out}_jh = index.div %{out}_j, %c2 : index
      %{out}_ql_a = index.mul %{out}_half, %{out}_c64 : index
      %{out}_ql_b = index.mul %{out}_jl, %c32 : index
      %{out}_ql_c = index.add %{out}_ql_a, %{out}_ql_b : index
      %{out}_ql_i = index.add %{out}_ql_c, %{out}_l0 : index
      %{out}_qh_a = index.mul %{out}_half, %c32 : index
      %{out}_qh_b = index.add %{out}_qh_base, %{out}_qh_a : index
      %{out}_qh_i = index.add %{out}_qh_b, %{out}_l0 : index
      %{out}_sc_a = index.mul %{out}_half, %c8 : index
      %{out}_sc_b = index.mul %{out}_j, %c2 : index
      %{out}_sc_c = index.div %{out}_l0, %c16 : index
      %{out}_sc_d = index.add %{out}_sc_a, %{out}_sc_b : index
      %{out}_sc_e = index.add %{out}_sc_d, %{out}_sc_c : index
      %{out}_sc_i = index.add %{out}_sc_base, %{out}_sc_e : index
      %{out}_block_add = index.scale %{out}_super, %block_bytes : index, offset -> offset
      %{out}_block_origin = index.add %w_row_origin, %{out}_block_add : offset
      %{out}_ql_off = index.cast %{out}_ql_i : index to offset
      %{out}_qh_off = index.cast %{out}_qh_i : index to offset
      %{out}_sc_off = index.cast %{out}_sc_i : index to offset
      %{out}_ql_o = index.add %{out}_block_origin, %{out}_ql_off : offset
      %{out}_qh_o = index.add %{out}_block_origin, %{out}_qh_off : offset
      %{out}_sc_o = index.add %{out}_block_origin, %{out}_sc_off : offset
      %{out}_d_o = index.add %{out}_block_origin, %{out}_d_bytes : offset
      %{out}_ql_v = buffer.view %{w}[%{out}_ql_o] : buffer -> view<16xi8>
      %{out}_qh_v = buffer.view %{w}[%{out}_qh_o] : buffer -> view<16xi8>
      %{out}_sc_v = buffer.view %{w}[%{out}_sc_o] : buffer -> view<1xi8>
      %{out}_d_v = buffer.view %{w}[%{out}_d_o] : buffer -> view<1xf16>
      %{out}_ql = vector.load %{out}_ql_v[0] : view<16xi8> -> vector<16xi8>
      %{out}_qh = vector.load %{out}_qh_v[0] : view<16xi8> -> vector<16xi8>
      %{out}_sc8 = view.load %{out}_sc_v[0] : view<1xi8> -> i8
      %{out}_dh = view.load %{out}_d_v[0] : view<1xf16> -> f16
      %{out}_s1 = index.mul %{out}_jh, %c4 : index
      %{out}_s2 = index.mul %{out}_j, %c2 : index
      %{out}_s1_32 = index.cast %{out}_s1 : index to i32
      %{out}_s2_32 = index.cast %{out}_s2 : index to i32
      %{out}_s1_8 = scalar.trunci %{out}_s1_32 : i32 to i8
      %{out}_s2_8 = scalar.trunci %{out}_s2_32 : i32 to i8
      %{out}_s1_v = vector.splat %{out}_s1_8 : vector<16xi8>
      %{out}_s2_v = vector.splat %{out}_s2_8 : vector<16xi8>
      %{out}_m15 = vector.constant 15 : vector<16xi8>
      %{out}_m3 = vector.constant 3 : vector<16xi8>
      %{out}_f4 = vector.constant 4 : vector<16xi8>
      %{out}_lo_s = vector.shrui %{out}_ql, %{out}_s1_v : vector<16xi8>
      %{out}_lo = vector.andi %{out}_lo_s, %{out}_m15 : vector<16xi8>
      %{out}_hi_s = vector.shrui %{out}_qh, %{out}_s2_v : vector<16xi8>
      %{out}_hi_m = vector.andi %{out}_hi_s, %{out}_m3 : vector<16xi8>
      %{out}_hi = vector.shli %{out}_hi_m, %{out}_f4 : vector<16xi8>
      %{out}_q = vector.ori %{out}_lo, %{out}_hi : vector<16xi8>
      %{out}_codes = vector.uitofp %{out}_q : vector<16xi8> to vector<16xf32>
      %{out}_bias = vector.constant 32.0 : vector<16xf32>
      %{out}_centered = vector.subf %{out}_codes, %{out}_bias : vector<16xf32>
      %{out}_d = scalar.extf %{out}_dh : f16 to f32
      %{out}_scf = scalar.sitofp %{out}_sc8 : i8 to f32
      %{out}_scale = scalar.mulf %{out}_d, %{out}_scf : f32
      %{out}_scales = vector.splat %{out}_scale : vector<16xf32>
      %{out}_scaled = vector.mulf %{out}_centered, %{out}_scales : vector<16xf32>
      %{out} = vector.fptrunc %{out}_scaled : vector<16xf32> to vector<16xf16>'''
    # Q4_K: 256-value superblocks of 144 bytes; 8 groups of 32 with 6-bit scale/min.
    return f'''
      %{out}_super = index.div %w_col, %c256 : index
      %{out}_within = index.rem %w_col, %c256 : index
      %{out}_group_value = index.div %{out}_within, %c32 : index
      %{out}_group = index.assume %{out}_group_value [range(%{out}_group_value, 0, 7)] : index
      %{out}_in_group = index.rem %{out}_within, %c32 : index
      %{out}_half = index.div %{out}_in_group, %c16 : index
      %{out}_block_add = index.scale %{out}_super, %block_bytes : index, offset -> offset
      %{out}_block_origin = index.add %w_row_origin, %{out}_block_add : offset
      %{out}_header_view = buffer.view %{w}[%{out}_block_origin] : buffer -> view<4xi32>
      %{out}_header = vector.load %{out}_header_view[0] : view<4xi32> -> vector<4xi32>
      %{out}_scale, %{out}_minimum = func.call {META}(%{out}_header, %{out}_group) : (vector<4xi32>, index) -> (f32, f32)
      %{out}_page = index.div %{out}_group, %c2 : index
      %{out}_nibble = index.rem %{out}_group, %c2 : index
      %{out}_word_page = index.mul %{out}_page, %c8 : index
      %{out}_word_half = index.mul %{out}_half, %c4 : index
      %{out}_word = index.add %{out}_word_page, %{out}_word_half : index
      %{out}_payload = index.add %{out}_block_origin, %sixteen : offset
      %{out}_words = buffer.view %{w}[%{out}_payload] : buffer -> view<32xi32>
      %{out}_packed = vector.load %{out}_words[%{out}_word] : view<32xi32> -> vector<4xi32>
      %{out}_shift_bits = index.mul %{out}_nibble, %c4 : index
      %{out}_shift_i32 = index.cast %{out}_shift_bits : index to i32
      %{out}_shift = vector.splat %{out}_shift_i32 : vector<4xi32>
      %{out}_shifted = vector.shrui %{out}_packed, %{out}_shift : vector<4xi32>
      %{out}_masked = vector.andi %{out}_shifted, %nibble_mask : vector<4xi32>
      %{out}_bytes = vector.bitcast %{out}_masked : vector<4xi32> to vector<16xi8>
      %{out}_codes = vector.uitofp %{out}_bytes : vector<16xi8> to vector<16xf16>
      %{out}_scale_h = scalar.fptrunc %{out}_scale : f32 to f16
      %{out}_neg_min = scalar.negf %{out}_minimum : f32
      %{out}_neg_min_h = scalar.fptrunc %{out}_neg_min : f32 to f16
      %{out}_scales = vector.splat %{out}_scale_h : vector<16xf16>
      %{out}_minima = vector.splat %{out}_neg_min_h : vector<16xf16>
      %{out} = vector.fmaf %{out}_codes, %{out}_scales, %{out}_minima : vector<16xf16>'''


def gemm(fmt, mode):
    f = FMT[fmt]
    two_w = mode == "swiglu"
    name = f"{PREFIX}_gemm_{fmt}_{mode}"
    wargs = "%w: buffer, %w2: buffer" if two_w else "%w: buffer"
    accs = ["g0", "g1", "u0", "u1"] if two_w else ["a0", "a1"]
    stages = ["w_stage", "w2_stage"] if two_w else ["w_stage"]
    k_min = 256 if fmt in ("q4_k", "q6_k") else 64
    lines = []
    lines.append(f'''kernel.def export("{name}") @{name}(%rows: index, %n: index) {{
  %c1 = index.constant 1 : index
  %c64 = index.constant 64 : index
  %c256 = index.constant 256 : index
  %row_tiles = index.div %rows, %c64 : index
  %out_tiles = index.div %n, %c64 : index
  kernel.launch.config workgroups(%row_tiles, %out_tiles, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%rows_arg: index, %k_arg: index, %n_arg: index, %x: buffer, %scales: buffer, {wargs}, %y: buffer) {{
  %rows = index.assume %rows_arg [range(%rows_arg, 64, 8192), multiple_of(%rows_arg, 64)] : index
  %k = index.assume %k_arg [range(%k_arg, {k_min}, 65536), multiple_of(%k_arg, {k_min})] : index
  %n = index.assume %n_arg [range(%n_arg, 64, 262144), multiple_of(%n_arg, 64)] : index
  %origin = index.constant 0 : offset
  %two = index.constant 2 : offset
  %sixteen = index.constant 16 : offset
  %block_bytes = index.constant {f["block_bytes"]} : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c8 = index.constant 8 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %c64 = index.constant 64 : index
  %c256 = index.constant 256 : index
  %nibble_mask = vector.constant 252645135 : vector<4xi32>
  %stage_bytes = index.constant 9216 : offset
  %row_tile_count = index.div %rows, %c64 : index
  %out_tile_count = index.div %n, %c64 : index
  %row_group_id = kernel.workgroup.id<x> : index
  %out_group_id = kernel.workgroup.id<y> : index
  %row_group = index.assume %row_group_id [lt(%row_group_id, %row_tile_count)] : index
  %out_group = index.assume %out_group_id [lt(%out_group_id, %out_tile_count)] : index
  %workitem_id = kernel.workitem.id<x> : index
  %workitem = index.assume %workitem_id [range(%workitem_id, 0, 255)] : index
  %subgroup = kernel.subgroup.id : index
  %lane = kernel.subgroup.lane.id : index
  %lane_row = index.rem %lane, %c16 : index
  %bounded_subgroup = index.assume %subgroup [range(%subgroup, 0, 7)] : index
  %wave_row_group = index.div %bounded_subgroup, %c2 : index
  %wave_out_group = index.rem %bounded_subgroup, %c2 : index
  %wave_row = index.mul %wave_row_group, %c16 : index
  %wave_out = index.mul %wave_out_group, %c32 : index
  %w_row0 = index.add %wave_out, %lane_row : index
  %w_row1 = index.add %w_row0, %c16 : index
  %row_origin = index.mul %row_group, %c64 : index
  %out_origin = index.mul %out_group, %c64 : index
  %result_row = index.add %row_origin, %wave_row : index
  %result_col0 = index.add %out_origin, %wave_out : index
  %result_col1 = index.add %result_col0, %c16 : index
  %blocks = index.div %k, %c{f["block_values"]} : index
  %row_bytes = index.scale %blocks, %block_bytes : index, offset -> offset
  %x_view = buffer.view %x[%origin] : buffer -> view<[%rows]x[%k]xf16>
  // 64x64 output tile with the full row stride n.
  %y_layout = encoding.layout.strided [%n, %c1] : encoding<layout>
  %tile_first = index.madd %row_origin, %n, %out_origin : index
  %four = index.constant 4 : offset
  %tile_offset = index.scale %tile_first, %four : index, offset -> offset
  %y_view = buffer.view %y[%tile_offset] : buffer -> view<64x64xf32, %y_layout>
  %bounded_wave_row = index.assume %wave_row [range(%wave_row, 0, 48)] : index
  %bounded_wave_out = index.assume %wave_out [range(%wave_out, 0, 32)] : index
  %wave_out1 = index.add %bounded_wave_out, %c16 : index
  %x_stage = buffer.alloca<workgroup> align(16) %stage_bytes : buffer
  %x_stage_view = buffer.view %x_stage[%origin] : buffer -> view<64x72xf16>''')
    for st in stages:
        lines.append(f'''  %{st} = buffer.alloca<workgroup> align(16) %stage_bytes : buffer
  %{st}_view = buffer.view %{st}[%origin] : buffer -> view<64x72xf16>''')
    lines.append('''  // Thread t stages 16 values: row t/4, columns (t%4)*16.
  %stage_row = index.div %workitem, %c4 : index
  %stage_quarter = index.rem %workitem, %c4 : index
  %stage_col = index.mul %stage_quarter, %c16 : index
  %x_row = index.add %row_origin, %stage_row : index
  %bounded_x_row = index.assume %x_row [lt(%x_row, %rows)] : index
  %w_out_row = index.add %out_origin, %stage_row : index
  %bounded_w_out_row = index.assume %w_out_row [lt(%w_out_row, %n)] : index
  %w_row_origin = index.scale %bounded_w_out_row, %row_bytes : index, offset -> offset
  %zero8 = vector.constant 0.0 : vector<8xf32>
  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>
  %k_steps = index.div %k, %c64 : index
  %s_view = buffer.view %scales[%origin] : buffer -> view<[%rows]x2xf32>
  %x_scale = view.load %s_view[%bounded_x_row, %c0] : view<[%rows]x2xf32> -> f32
  %x_scales = vector.splat %x_scale : vector<16xf32>''')
    acc_args = ", ".join(f"%{a}_acc = %init : vector<8xf32>" for a in accs)
    acc_types = ", ".join("vector<8xf32>" for _ in accs)
    lines.append(f'''  {", ".join("%" + a + "_raw" for a in accs)} = scf.for %step = [%c0 to %k_steps step %c1]({acc_args}) -> ({acc_types}) {{
    %k_origin = index.mul %step, %c64 : index
    %x_col = index.add %k_origin, %stage_col : index
    %x_half = vector.load %x_view[%bounded_x_row, %x_col] : view<[%rows]x[%k]xf16> -> vector<16xf16>
    vector.store %x_half, %x_stage_view[%stage_row, %stage_col] : vector<16xf16>, view<64x72xf16>
    %w_col = index.add %k_origin, %stage_col : index''')
    for i, (st, w) in enumerate(zip(stages, ["w", "w2"])):
        lines.append(dequant16(fmt, w, None, f"deq{i}"))
        lines.append(f"    vector.store %deq{i}, %{st}_view[%stage_row, %stage_col] : vector<16xf16>, view<64x72xf16>")
    lines.append("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    cur = {a: f"%{a}_acc" for a in accs}
    for kk in range(4):
        col = kk * 16
        lines.append(f'''    %fc{kk} = index.constant {col} : index
    %lhs{kk} = vector.fragment.load<lhs> %x_stage_view[%wave_row, %fc{kk}] shape [%c16, %c16] : view<64x72xf16> -> vector<16xf16>''')
        for si, st in enumerate(stages):
            pre = "g" if (two_w and si == 0) else ("u" if two_w else "a")
            for r in range(2):
                wr = f"%w_row{r}"
                lines.append(f'''    %rv{si}{r}{kk} = vector.load %{st}_view[{wr}, %fc{kk}] : view<64x72xf16> -> vector<16xf16>
    %rhs{si}{r}{kk} = vector.fragment<rhs> %rv{si}{r}{kk} shape [%c16, %c16] : vector<16xf16>
    %n{pre}{r}_{kk} = vector.mma %lhs{kk}, %rhs{si}{r}{kk}, {cur[pre + str(r)]} : vector<16xf16>, vector<16xf16>, vector<8xf32>''')
                cur[pre + str(r)] = f"%n{pre}{r}_{kk}"
    lines.append("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    lines.append(f"    scf.yield {', '.join(cur[a] for a in accs)} : {acc_types}\n  }}")
    lines.append('''  %inv_bytes = index.constant 4096 : offset
  %inv_tile = buffer.alloca<workgroup> align(16) %inv_bytes : buffer
  %inv_view = buffer.view %inv_tile[%origin] : buffer -> view<64x16xf32>
  %writer = index.cmp ult, %workitem, %c64 : index
  scf.if %writer {
    %inv_row_value = index.add %row_origin, %workitem : index
    %inv_row = index.assume %inv_row_value [lt(%inv_row_value, %rows)] : index
    %inverse = view.load %s_view[%inv_row, %c1] : view<[%rows]x2xf32> -> f32
    %inverse16 = vector.splat %inverse : vector<16xf32>
    %bw = index.assume %workitem [range(%workitem, 0, 63)] : index
    vector.store %inverse16, %inv_view[%bw, %c0] : vector<16xf32>, view<64x16xf32>
  }
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %inv_frag = vector.fragment.load<result> %inv_view[%wave_row, %c0] shape [%c16, %c16] : view<64x16xf32> -> vector<8xf32>''')
    raw = {a: a + "_raw" for a in accs}
    for a in accs:
        lines.append(f"  %{a} = vector.mulf %{a}_raw, %inv_frag : vector<8xf32>")
    if mode == "store":
        lines.append('''  vector.fragment.store<result> %a0, %y_view[%bounded_wave_row, %bounded_wave_out] shape [%c16, %c16] : vector<8xf32>, view<64x64xf32, %y_layout>
  vector.fragment.store<result> %a1, %y_view[%bounded_wave_row, %wave_out1] shape [%c16, %c16] : vector<8xf32>, view<64x64xf32, %y_layout>''')
    elif mode == "acc":
        lines.append('''  %old0 = vector.fragment.load<result> %y_view[%bounded_wave_row, %bounded_wave_out] shape [%c16, %c16] : view<64x64xf32, %y_layout> -> vector<8xf32>
  %old1 = vector.fragment.load<result> %y_view[%bounded_wave_row, %wave_out1] shape [%c16, %c16] : view<64x64xf32, %y_layout> -> vector<8xf32>
  %sum0 = vector.addf %old0, %a0 : vector<8xf32>
  %sum1 = vector.addf %old1, %a1 : vector<8xf32>
  vector.fragment.store<result> %sum0, %y_view[%bounded_wave_row, %bounded_wave_out] shape [%c16, %c16] : vector<8xf32>, view<64x64xf32, %y_layout>
  vector.fragment.store<result> %sum1, %y_view[%bounded_wave_row, %wave_out1] shape [%c16, %c16] : vector<8xf32>, view<64x64xf32, %y_layout>''')
    else:
        lines.append('''  %sg0 = vector.siluf<afn> %g0 : vector<8xf32>
  %sg1 = vector.siluf<afn> %g1 : vector<8xf32>
  %act0 = vector.mulf %sg0, %u0 : vector<8xf32>
  %act1 = vector.mulf %sg1, %u1 : vector<8xf32>
  vector.fragment.store<result> %act0, %y_view[%bounded_wave_row, %bounded_wave_out] shape [%c16, %c16] : vector<8xf32>, view<64x64xf32, %y_layout>
  vector.fragment.store<result> %act1, %y_view[%bounded_wave_row, %wave_out1] shape [%c16, %c16] : vector<8xf32>, view<64x64xf32, %y_layout>''')
    lines.append("  kernel.return\n}\n")
    return "\n".join(lines)


def gemm_big(fmt, mode, tn=None, suffix=""):
    """128-row WMMA GEMM. store/acc: 128x128 tiles, waves 4 (32 rows) x 2 (64 cols).
    swiglu: 128x64 tiles per weight, waves 4 x 2 (32 cols each of gate and up)."""
    f = FMT[fmt]
    two_w = mode == "swiglu"
    TN = tn or (64 if two_w else 128)
    wave_cols = TN // 2
    cfr = wave_cols // 16
    name = f"{PREFIX}_gemm_{fmt}_{mode}_m128" + suffix
    weights = ["w", "w2"] if two_w else ["w"]
    k_min = 256 if fmt in ("q4_k", "q6_k") else 64
    L = []
    L.append(f'''kernel.def export("{name}") @{name}(%rows: index, %n: index) {{
  %c1 = index.constant 1 : index
  %c128 = index.constant 128 : index
  %tn = index.constant {TN} : index
  %c256 = index.constant 256 : index
  %row_tiles = index.div %rows, %c128 : index
  %out_tiles = index.div %n, %tn : index
  kernel.launch.config workgroups(%row_tiles, %out_tiles, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%rows_arg: index, %k_arg: index, %n_arg: index, %x: buffer, %scales: buffer, {", ".join(f"%{w}: buffer" for w in weights)}, %y: buffer) {{
  %rows = index.assume %rows_arg [range(%rows_arg, 128, 8192), multiple_of(%rows_arg, 128)] : index
  %k = index.assume %k_arg [range(%k_arg, {k_min}, 65536), multiple_of(%k_arg, {k_min})] : index
  %n = index.assume %n_arg [range(%n_arg, {TN}, 262144), multiple_of(%n_arg, {TN})] : index
  %origin = index.constant 0 : offset
  %two = index.constant 2 : offset
  %sixteen = index.constant 16 : offset
  %four = index.constant 4 : offset
  %block_bytes = index.constant {f["block_bytes"]} : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c8 = index.constant 8 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %c64 = index.constant 64 : index
  %c128 = index.constant 128 : index
  %c256 = index.constant 256 : index
  %tn = index.constant {TN} : index
  %nibble_mask = vector.constant 252645135 : vector<4xi32>
  %row_tile_count = index.div %rows, %c128 : index
  %out_tile_count = index.div %n, %tn : index
  %row_group_id = kernel.workgroup.id<x> : index
  %out_group_id = kernel.workgroup.id<y> : index
  %row_group = index.assume %row_group_id [lt(%row_group_id, %row_tile_count)] : index
  %out_group = index.assume %out_group_id [lt(%out_group_id, %out_tile_count)] : index
  %workitem_id = kernel.workitem.id<x> : index
  %workitem = index.assume %workitem_id [range(%workitem_id, 0, 255)] : index
  %subgroup = kernel.subgroup.id : index
  %bounded_subgroup = index.assume %subgroup [range(%subgroup, 0, 7)] : index
  %lane = kernel.subgroup.lane.id : index
  %lane_row = index.rem %lane, %c16 : index
  %wave_row_group = index.div %bounded_subgroup, %c2 : index
  %wave_col_group = index.rem %bounded_subgroup, %c2 : index
  %wave_row_value = index.mul %wave_row_group, %c32 : index
  %wave_row = index.assume %wave_row_value [range(%wave_row_value, 0, 96)] : index
  %wave_row1 = index.add %wave_row, %c16 : index
  %wave_cols = index.constant {wave_cols} : index
  %wave_col_value = index.mul %wave_col_group, %wave_cols : index
  %wave_col = index.assume %wave_col_value [range(%wave_col_value, 0, {TN - wave_cols})] : index
  %row_origin = index.mul %row_group, %c128 : index
  %out_origin = index.mul %out_group, %tn : index
  %blocks = index.div %k, %c{f["block_values"]} : index
  %row_bytes = index.scale %blocks, %block_bytes : index, offset -> offset
  %x_view = buffer.view %x[%origin] : buffer -> view<[%rows]x[%k]xf16>
  %s_view = buffer.view %scales[%origin] : buffer -> view<[%rows]x2xf32>
  %x_bytes = index.constant 18432 : offset
  %x_stage = buffer.alloca<workgroup> align(16) %x_bytes : buffer
  %x_stage_view = buffer.view %x_stage[%origin] : buffer -> view<128x72xf16>
  %w_bytes = index.constant {TN * 144} : offset''')
    for w in weights:
        L.append(f'''  %{w}_stage = buffer.alloca<workgroup> align(16) %w_bytes : buffer
  %{w}_stage_view = buffer.view %{w}_stage[%origin] : buffer -> view<{TN}x72xf16>''')
    per_thread_w = TN * 64 // 256
    L.append(f'''  // A: thread t stages 32 values of row t/2. W: {per_thread_w} values of row t/{256 // TN}.
  %x_stage_row = index.div %workitem, %c2 : index
  %x_half = index.rem %workitem, %c2 : index
  %x_stage_col = index.mul %x_half, %c32 : index
  %x_row_value = index.add %row_origin, %x_stage_row : index
  %x_row = index.assume %x_row_value [lt(%x_row_value, %rows)] : index
  %x_scale = view.load %s_view[%x_row, %c0] : view<[%rows]x2xf32> -> f32
  %x_scales = vector.splat %x_scale : vector<32xf32>
  %w_per_row = index.constant {256 // TN} : index
  %stage_row = index.div %workitem, %w_per_row : index
  %w_seg = index.rem %workitem, %w_per_row : index
  %w_seg_width = index.constant {per_thread_w} : index
  %stage_col = index.mul %w_seg, %w_seg_width : index
  %w_out_row_value = index.add %out_origin, %stage_row : index
  %w_out_row = index.assume %w_out_row_value [lt(%w_out_row_value, %n)] : index
  %w_row_origin = index.scale %w_out_row, %row_bytes : index, offset -> offset
  %zero8 = vector.constant 0.0 : vector<8xf32>
  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>
  %k_steps = index.div %k, %c64 : index''')
    accs = [f"{wi}r{r}c{c}" for wi in range(len(weights)) for r in range(2) for c in range(cfr)]
    import re as _re
    # Register prefetch: each weight segment's dequant splits into an address +
    # load part (issued for step+1 before this step's MMAs) and a decode part
    # that consumes the raw values carried through the loop.
    segs = []
    for wi, w in enumerate(weights):
        for seg in range(per_thread_w // 16):
            out = f"dq{wi}_{seg}"
            full = dequant16(fmt, w, None, out).replace("%w_col,", f"%wcol_{wi}_{seg},").replace("%w_col :", f"%wcol_{wi}_{seg} :")
            lines = [l for l in full.strip("\n").split("\n") if l.strip()]
            last = max(i for i, l in enumerate(lines) if ".load " in l)
            addr = lines[: last + 1]
            loads = [(_re.match(r"\s*(%\S+) = ", l).group(1), l.split("-> ")[-1].strip()) for l in addr if ".load " in l]
            segs.append((wi, seg, w, out, lines, addr, loads))
    def addr_text(sfx, kvar):
        t = []
        for wi, seg, w, out, lines, addr, loads in segs:
            t.append(f"    %wc{wi}_{seg}_c{sfx} = index.constant {seg * 16} : index\n    %wc{wi}_{seg}_l{sfx} = index.add %stage_col, %wc{wi}_{seg}_c{sfx} : index\n    %wcol_{wi}_{seg}{sfx} = index.add {kvar}, %wc{wi}_{seg}_l{sfx} : index")
            body = "\n".join(addr).replace(f"%wcol_{wi}_{seg}", f"%wcol_{wi}_{seg}{sfx}")
            body = _re.sub(rf"%{out}_", f"%{out}{sfx}_", body)
            t.append(body)
        if sfx == "p":
            t.append(f"    %x_col{sfx} = index.add {kvar}, %x_stage_col : index\n    %x_raw{sfx} = vector.load %x_view[%x_row, %x_col{sfx}] : view<[%rows]x[%k]xf16> -> vector<32xf16>")
        else:
            t.append(f"""    %x_more = index.cmp ult, %x_next_step, %k_steps : index
    %x_raw{sfx} = scf.if %x_more -> (vector<32xf16>) {{
      %x_ns = index.assume %x_next_step [lt(%x_next_step, %k_steps)] : index
      %x_ko = index.mul %x_ns, %c64 : index
      %x_col{sfx} = index.add %x_ko, %x_stage_col : index
      %xv = vector.load %x_view[%x_row, %x_col{sfx}] : view<[%rows]x[%k]xf16> -> vector<32xf16>
      scf.yield %xv : vector<32xf16>
    }} else {{
      scf.yield %cr_x : vector<32xf16>
    }}""")
        return "\n".join(t)
    carried = []   # (iter name, init name, next name, type)
    for wi, seg, w, out, lines, addr, loads in segs:
        for name, ty in loads:
            base = name[1:]
            carried.append((f"%cr_{base}", f"%{base.replace(out + '_', out + 'p_', 1)}", f"%{base.replace(out + '_', out + 'n_', 1)}", ty))
    carried.append(("%cr_x", "%x_rawp", "%x_rawn", "vector<32xf16>"))
    L.append(addr_text("p", "%c0").replace("    ", "  ", 1))
    L.append("  %k_last = index.sub %k_steps, %c1 : index")
    L.append("  " + ", ".join(f"%o{a}" for a in accs) + ", " + ", ".join(f"%unused_{i}" for i in range(len(carried))) + " = scf.for %step = [%c0 to %k_steps step %c1](" +
             ", ".join(f"%i{a} = %init : vector<8xf32>" for a in accs) + ", " + ", ".join(f"{c[0]} = {c[1]} : {c[3]}" for c in carried) + ") -> (" +
             ", ".join("vector<8xf32>" for _ in accs) + ", " + ", ".join(c[3] for c in carried) + ") {")
    L.append("    %k_origin = index.mul %step, %c64 : index\n    vector.store %cr_x, %x_stage_view[%x_stage_row, %x_stage_col] : vector<32xf16>, view<128x72xf16>")
    for wi, seg, w, out, lines, addr, loads in segs:
        L.append(f"    %wc{wi}_{seg}_c = index.constant {seg * 16} : index\n    %wc{wi}_{seg}_l = index.add %stage_col, %wc{wi}_{seg}_c : index\n    %wcol_{wi}_{seg} = index.add %k_origin, %wc{wi}_{seg}_l : index")
        body = [l for l in lines if ".load " not in l]
        text = "\n".join(body)
        for name, ty in loads:
            text = _re.sub(_re.escape(name) + r"\b", f"%cr_{name[1:]}", text)
        L.append(text)
        L.append(f"    vector.store %{out}, %{w}_stage_view[%stage_row, %wc{wi}_{seg}_l] : vector<16xf16>, view<{TN}x72xf16>")
    L.append("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    L.append("    %k_next_raw = index.add %k_origin, %c64 : index\n    %k_limit = index.sub %k, %c64 : index\n    %k_next = index.min %k_next_raw, %k_limit : index\n    %x_next_step = index.add %step, %c1 : index")
    L.append(addr_text("n", "%k_next"))
    cur = {a: f"%i{a}" for a in accs}
    for kk in range(4):
        L.append(f'''    %fk{kk} = index.constant {kk * 16} : index
    %l0_{kk} = vector.fragment.load<lhs> %x_stage_view[%wave_row, %fk{kk}] shape [%c16, %c16] : view<128x72xf16> -> vector<16xf16>
    %l1_{kk} = vector.fragment.load<lhs> %x_stage_view[%wave_row1, %fk{kk}] shape [%c16, %c16] : view<128x72xf16> -> vector<16xf16>''')
        for wi, w in enumerate(weights):
            for c in range(cfr):
                L.append(f'''    %wr{wi}{c}_{kk}_c = index.constant {c * 16} : index
    %wr{wi}{c}_{kk}_b = index.add %wave_col, %wr{wi}{c}_{kk}_c : index
    %wr{wi}{c}_{kk} = index.add %wr{wi}{c}_{kk}_b, %lane_row : index
    %rv{wi}{c}_{kk} = vector.load %{w}_stage_view[%wr{wi}{c}_{kk}, %fk{kk}] : view<{TN}x72xf16> -> vector<16xf16>
    %rf{wi}{c}_{kk} = vector.fragment<rhs> %rv{wi}{c}_{kk} shape [%c16, %c16] : vector<16xf16>''')
                for r in range(2):
                    a = f"{wi}r{r}c{c}"
                    L.append(f"    %m{a}_{kk} = vector.mma %l{r}_{kk}, %rf{wi}{c}_{kk}, {cur[a]} : vector<16xf16>, vector<16xf16>, vector<8xf32>")
                    cur[a] = f"%m{a}_{kk}"
    L.append("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    L.append("    scf.yield " + ", ".join(cur[a] for a in accs) + ", " + ", ".join(c[2] for c in carried) + " : " + ", ".join("vector<8xf32>" for _ in accs) + ", " + ", ".join(c[3] for c in carried) + "\n  }")
    L.append(f'''  %inv_bytes = index.constant 8192 : offset
  %inv_tile = buffer.alloca<workgroup> align(16) %inv_bytes : buffer
  %inv_view = buffer.view %inv_tile[%origin] : buffer -> view<128x16xf32>
  %writer = index.cmp ult, %workitem, %c128 : index
  scf.if %writer {{
    %bw = index.assume %workitem [range(%workitem, 0, 127)] : index
    %inv_row_value = index.add %row_origin, %bw : index
    %inv_row = index.assume %inv_row_value [lt(%inv_row_value, %rows)] : index
    %inverse = view.load %s_view[%inv_row, %c1] : view<[%rows]x2xf32> -> f32
    %inverse16 = vector.splat %inverse : vector<16xf32>
    vector.store %inverse16, %inv_view[%bw, %c0] : vector<16xf32>, view<128x16xf32>
  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %inv0 = vector.fragment.load<result> %inv_view[%wave_row, %c0] shape [%c16, %c16] : view<128x16xf32> -> vector<8xf32>
  %inv1 = vector.fragment.load<result> %inv_view[%wave_row1, %c0] shape [%c16, %c16] : view<128x16xf32> -> vector<8xf32>
  %y_layout = encoding.layout.strided [%n, %c1] : encoding<layout>
  %tile_first = index.madd %row_origin, %n, %out_origin : index
  %tile_offset = index.scale %tile_first, %four : index, offset -> offset
  %y_view = buffer.view %y[%tile_offset] : buffer -> view<128x{TN}xf32, %y_layout>''')
    for r in range(2):
        for c in range(cfr):
            row = "%wave_row" if r == 0 else "%wave_row1"
            L.append(f"  %oc{r}{c}_c = index.constant {c * 16} : index\n  %oc{r}{c} = index.add %wave_col, %oc{r}{c}_c : index")
            if two_w:
                L.append(f'''  %g{r}{c} = vector.mulf %o0r{r}c{c}, %inv{r} : vector<8xf32>
  %u{r}{c} = vector.mulf %o1r{r}c{c}, %inv{r} : vector<8xf32>
  %sg{r}{c} = vector.siluf<afn> %g{r}{c} : vector<8xf32>
  %v{r}{c} = vector.mulf %sg{r}{c}, %u{r}{c} : vector<8xf32>''')
            else:
                L.append(f"  %v{r}{c}_s = vector.mulf %o0r{r}c{c}, %inv{r} : vector<8xf32>")
                if mode == "acc":
                    L.append(f"  %old{r}{c} = vector.fragment.load<result> %y_view[{row}, %oc{r}{c}] shape [%c16, %c16] : view<128x{TN}xf32, %y_layout> -> vector<8xf32>\n  %v{r}{c} = vector.addf %old{r}{c}, %v{r}{c}_s : vector<8xf32>")
                else:
                    L.append(f"  %v{r}{c} = vector.addf %v{r}{c}_s, %zero8 : vector<8xf32>")
            L.append(f"  vector.fragment.store<result> %v{r}{c}, %y_view[{row}, %oc{r}{c}] shape [%c16, %c16] : vector<8xf32>, view<128x{TN}xf32, %y_layout>")
    L.append("  kernel.return\n}\n")
    return "\n".join(L)


if __name__ == "__main__":
    out = [f"// Generated by gen_gemm.py: {PREFIX} prefill WMMA GEMMs over GGUF weights.\n"]
    spec = os.environ.get("GEMM_FORMATS", "q4_k:store+acc+swiglu,q6_k:store")
    def skip_padding(text):
        # GEMM_SKIP_PADDING: a row tile whose first row has scale 0 (a padding
        # row, as marked by the row-scale pass) does no work.
        anchor = "  %s_view = buffer.view %scales[%origin] : buffer -> view<[%rows]x2xf32>\n"
        assert anchor in text
        check = anchor + """  %tile_row = index.assume %row_origin [lt(%row_origin, %rows)] : index
  %tile_scale = view.load %s_view[%tile_row, %c0] : view<[%rows]x2xf32> -> f32
  %no_scale = scalar.constant 0.0 : f32
  %tile_live = scalar.cmpf one, %tile_scale, %no_scale : f32
  scf.if %tile_live {
"""
        text = text.replace(anchor, check, 1)
        tail = "  kernel.return\n}\n"
        assert text.endswith(tail)
        return text[: -len(tail)] + "  }\n" + tail
    def n_fastest(text):
        # GEMM_N_FASTEST: walk output-column tiles fastest so consecutive
        # workgroups share an A row tile (kept in L2 instead of re-read from DRAM).
        text = text.replace("workgroups(%row_tiles, %out_tiles, %c1)", "workgroups(%out_tiles, %row_tiles, %c1)")
        text = text.replace("  %row_group_id = kernel.workgroup.id<x> : index\n  %out_group_id = kernel.workgroup.id<y> : index",
                            "  %row_group_id = kernel.workgroup.id<y> : index\n  %out_group_id = kernel.workgroup.id<x> : index")
        return text
    on = lambda k: os.environ.get(k, "0") not in ("", "0", "false", "no")  # GEMM_X=0 really means off
    skip = skip_padding if on("GEMM_SKIP_PADDING") else (lambda t: t)
    order = n_fastest if on("GEMM_N_FASTEST") else (lambda t: t)
    post = lambda t: order(skip(t))
    for item in spec.split(","):
        fmt, modes = item.split(":")
        for mode in modes.split("+"):
            out.append(post(gemm(fmt, mode)))
            out.append(post(gemm_big(fmt, mode)))
    sys.stdout.write("\n".join(out))
