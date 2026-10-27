"""Generates Flint prefill MoE kernels (grouped WMMA GEMMs over routed experts).

Assignments a = token * 6 + slot are bucketed by expert into 32-row tiles.
tile_info[tile] = (expert, rows) with expert -1 for unused tiles; rowmap
gives each padded tile row's assignment (or -1); pos gives each assignment's
padded row. Gate/up writes act_sorted[padded_row, 512]; down writes
y_sorted[padded_row, 2048]; combine adds the weighted rows into the residual.
"""
import sys
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from gen_gemm import dequant16

BUCKET = r"""
// Bucket the tokens' six routed experts into 32-row tiles. One workgroup with
// workgroup-memory counters: lanes stride over assignments, so the cost is
// 6T/256 atomics per lane. Row order within an expert is unspecified; rows are
// independent, so results do not depend on it.
kernel.def export("flint_moe_bucket") @flint_moe_bucket(%tokens: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%tokens_arg: index, %max_tiles_arg: index, %info: buffer, %ids: buffer, %tile_info: buffer, %rowmap: buffer, %pos: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %max_tiles = index.assume %max_tiles_arg [range(%max_tiles_arg, 1, 65536)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c6 = index.constant 6 : index
  %c32 = index.constant 32 : index
  %c256 = index.constant 256 : index
  %c0_i32 = scalar.constant 0 : i32
  %c1_i32 = scalar.constant 1 : i32
  %c5_i32 = scalar.constant 5 : i32
  %c31_i32 = scalar.constant 31 : i32
  %c32_i32 = scalar.constant 32 : i32
  %minus_one = scalar.constant -1 : i32
  %lane32 = index.cast %lane : index to i32
  %assignments = index.mul %tokens, %c6 : index
  %ids_view = buffer.view %ids[%zero] : buffer -> view<[%tokens]x8xi32>
  %token_info = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %c2_info = index.constant 2 : index
  %counter_bytes = index.constant 2048 : offset
  %counters = buffer.alloca<workgroup> align(16) %counter_bytes : buffer
  %count_view = buffer.view %counters[%zero] : buffer -> view<256xi32>
  %cursor_origin = index.constant 1024 : offset
  %cursor_view = buffer.view %counters[%cursor_origin] : buffer -> view<256xi32>
  view.store %c0_i32, %count_view[%lane] : i32, view<256xi32>
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  scf.for %a = [%lane to %assignments step %c256] {
    %t_value = index.div %a, %c6 : index
    %t = index.assume %t_value [lt(%t_value, %tokens)] : index
    %s_value = index.rem %a, %c6 : index
    %s = index.assume %s_value [range(%s_value, 0, 5)] : index
    %id = view.load %ids_view[%t, %s] : view<[%tokens]x8xi32> -> i32
    %low = scalar.maxsi %id, %c0_i32 : i32
    %c255_i32 = scalar.constant 255 : i32
    %clamped = scalar.minsi %low, %c255_i32 : i32
    %bounded = scalar.assume %clamped [range(%clamped, 0, 255)] : i32
    %e = index.cast %bounded : i32 to index
    %live32 = view.load %token_info[%t, %c2_info] : view<[%tokens]x4xi32> -> i32
    %live = scalar.cmpi ne, %live32, %c0_i32 : i32
    scf.if %live {
      %old = view.atomic.rmw<addi> %c1_i32, %count_view[%e] {ordering = relaxed, scope = workgroup} : i32, view<256xi32> -> i32
    }
  }
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %count32 = view.load %count_view[%lane] : view<256xi32> -> i32
  %padded = scalar.addi %count32, %c31_i32 : i32
  %tiles32 = scalar.shrui %padded, %c5_i32 : i32
  %base32 = kernel.workgroup.scan<addi> %tiles32 {direction = forward, mode = exclusive} : i32
  %total32 = kernel.workgroup.reduce<addi> %tiles32 : i32
  %info_view = buffer.view %tile_info[%zero] : buffer -> view<[%max_tiles]x2xi32>
  %rowmap_rows = index.mul %max_tiles, %c32 : index
  %rowmap_view = buffer.view %rowmap[%zero] : buffer -> view<[%rowmap_rows]xi32>
  %pos_view = buffer.view %pos[%zero] : buffer -> view<[%assignments]xi32>
  %base_bound = scalar.assume %base32 [range(%base32, 0, 65535)] : i32
  %base = index.cast %base_bound : i32 to index
  %tiles_bound = scalar.assume %tiles32 [range(%tiles32, 0, 65535)] : i32
  %tiles = index.cast %tiles_bound : i32 to index
  %count_bound = scalar.assume %count32 [range(%count32, 0, 2097152)] : i32
  %count = index.cast %count_bound : i32 to index
  scf.for %j = [%c0 to %tiles step %c1] {
    %tile_value = index.add %base, %j : index
    %tile = index.assume %tile_value [lt(%tile_value, %max_tiles)] : index
    %first = index.mul %j, %c32 : index
    %remaining = index.sub %count, %first : index
    %full = index.cmp uge, %remaining, %c32 : index
    %rows = scf.select %full, %c32, %remaining : index
    %rows32 = index.cast %rows : index to i32
    view.store %lane32, %info_view[%tile, %c0] : i32, view<[%max_tiles]x2xi32>
    view.store %rows32, %info_view[%tile, %c1] : i32, view<[%max_tiles]x2xi32>
  }
  %total_bound = scalar.assume %total32 [range(%total32, 0, 65535)] : i32
  %total = index.cast %total_bound : i32 to index
  %unused_start = index.add %total, %lane : index
  scf.for %u = [%unused_start to %max_tiles step %c256] {
    %bu = index.assume %u [lt(%u, %max_tiles)] : index
    view.store %minus_one, %info_view[%bu, %c0] : i32, view<[%max_tiles]x2xi32>
    view.store %c0_i32, %info_view[%bu, %c1] : i32, view<[%max_tiles]x2xi32>
  }
  %first_row = index.mul %base, %c32 : index
  %first_row32 = index.cast %first_row : index to i32
  view.store %first_row32, %cursor_view[%lane] : i32, view<256xi32>
  // Padded tail rows of this expert's last tile carry no assignment.
  %used_end = index.add %first_row, %count : index
  %tile_rows = index.mul %tiles, %c32 : index
  %end_row = index.add %first_row, %tile_rows : index
  scf.for %r = [%used_end to %end_row step %c1] {
    %row = index.assume %r [range(%r, 0, 2097151), lt(%r, %rowmap_rows)] : index
    view.store %minus_one, %rowmap_view[%row] : i32, view<[%rowmap_rows]xi32>
  }
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  scf.for %a = [%lane to %assignments step %c256] {
    %t_value = index.div %a, %c6 : index
    %t = index.assume %t_value [lt(%t_value, %tokens)] : index
    %s_value = index.rem %a, %c6 : index
    %s = index.assume %s_value [range(%s_value, 0, 5)] : index
    %id = view.load %ids_view[%t, %s] : view<[%tokens]x8xi32> -> i32
    %low = scalar.maxsi %id, %c0_i32 : i32
    %c255_i32 = scalar.constant 255 : i32
    %clamped = scalar.minsi %low, %c255_i32 : i32
    %bounded = scalar.assume %clamped [range(%clamped, 0, 255)] : i32
    %e = index.cast %bounded : i32 to index
    %live32p = view.load %token_info[%t, %c2_info] : view<[%tokens]x4xi32> -> i32
    %livep = scalar.cmpi ne, %live32p, %c0_i32 : i32
    scf.if %livep {
    %row32 = view.atomic.rmw<addi> %c1_i32, %cursor_view[%e] {ordering = relaxed, scope = workgroup} : i32, view<256xi32> -> i32
    %row_bound = scalar.assume %row32 [range(%row32, 0, 2097151)] : i32
    %row_value = index.cast %row_bound : i32 to index
    %row = index.assume %row_value [lt(%row_value, %rowmap_rows)] : index
    %ba = index.assume %a [lt(%a, %assignments)] : index
    %a32 = index.cast %a : index to i32
    view.store %a32, %rowmap_view[%row] : i32, view<[%rowmap_rows]xi32>
    view.store %row32, %pos_view[%ba] : i32, view<[%assignments]xi32>
    }
  }
  kernel.return
}

// x[t] += sum_slot weight[t, slot] * y_sorted[pos[t * 6 + slot]].
kernel.def export("flint_moe_combine") @flint_moe_combine(%tokens: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%tokens_arg: index, %max_rows_arg: index, %info: buffer, %weights: buffer, %pos: buffer, %y_sorted: buffer, %x: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %max_rows = index.assume %max_rows_arg [range(%max_rows_arg, 1, 2097152)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c6 = index.constant 6 : index
  %assignments = index.mul %tokens, %c6 : index
  %w_view = buffer.view %weights[%zero] : buffer -> view<[%tokens]x8xf32>
  %pos_view = buffer.view %pos[%zero] : buffer -> view<[%assignments]xi32>
  %y_view = buffer.view %y_sorted[%zero] : buffer -> view<[%max_rows]x256x8xf32>
  %x_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x256x8xf32>
  %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %c2_info = index.constant 2 : index
  %live32 = view.load %info_view[%token, %c2_info] : view<[%tokens]x4xi32> -> i32
  %c0_live = scalar.constant 0 : i32
  %live = scalar.cmpi ne, %live32, %c0_live : i32
  scf.if %live {
  %old = vector.load %x_view[%token, %lane, %c0] : view<[%tokens]x256x8xf32> -> vector<8xf32>
  %total = scf.for %s = [%c0 to %c6 step %c1](%acc = %old : vector<8xf32>) -> (vector<8xf32>) {
    %bs = index.assume %s [range(%s, 0, 5)] : index
    %a_value = index.madd %token, %c6, %bs : index
    %a = index.assume %a_value [lt(%a_value, %assignments)] : index
    %row32 = view.load %pos_view[%a] : view<[%assignments]xi32> -> i32
    %row_bound = scalar.assume %row32 [range(%row32, 0, 2097151)] : i32
    %row_value = index.cast %row_bound : i32 to index
    %row = index.assume %row_value [lt(%row_value, %max_rows)] : index
    %w = view.load %w_view[%token, %bs] : view<[%tokens]x8xf32> -> f32
    %ws = vector.splat %w : vector<8xf32>
    %yv = vector.load %y_view[%row, %lane, %c0] : view<[%max_rows]x256x8xf32> -> vector<8xf32>
    %next = vector.fmaf %ws, %yv, %acc : vector<8xf32>
    scf.yield %next : vector<8xf32>
  }
  vector.store %total, %x_view[%token, %lane, %c0] : vector<8xf32>, view<[%tokens]x256x8xf32>
  }
  kernel.return
}
"""


def grouped(kind):
    """kind 'gate_up': A = h rows gathered via rowmap (K=2048), two Q4_K weights, SwiGLU -> act_sorted (N=512).
       kind 'down': A = act_sorted rows (K=512), one Q4_K weight -> y_sorted (N=2048)."""
    gate_up = kind == "gate_up"
    k = 2048 if gate_up else 512
    n = 512 if gate_up else 2048
    name = f"flint_moe_{kind}_grouped"
    wargs = "%w: buffer, %w2: buffer" if gate_up else "%w: buffer"
    stages = ["w_stage", "w2_stage"] if gate_up else ["w_stage"]
    row_bytes = (k // 256) * 144
    L = []
    L.append(f'''kernel.def export("{name}") @{name}(%max_tiles: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %out_tiles = index.constant {n // 64} : index
  kernel.launch.config workgroups(%max_tiles, %out_tiles, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%max_tiles_arg: index, %tokens_arg: index, %a: buffer, %scales: buffer, %tile_info: buffer, %rowmap: buffer, {wargs}, %y: buffer) {{
  %max_tiles = index.assume %max_tiles_arg [range(%max_tiles_arg, 1, 65536)] : index
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %origin = index.constant 0 : offset
  %sixteen = index.constant 16 : offset
  %four = index.constant 4 : offset
  %block_bytes = index.constant 144 : offset
  %expert_bytes = index.constant 589824 : offset
  %w_row_bytes = index.constant {row_bytes} : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c6 = index.constant 6 : index
  %c8 = index.constant 8 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %c64 = index.constant 64 : index
  %c256 = index.constant 256 : index
  %k = index.constant {k} : index
  %n = index.constant {n} : index
  %c0_i32 = scalar.constant 0 : i32
  %nibble_mask = vector.constant 252645135 : vector<4xi32>
  %tile_id = kernel.workgroup.id<x> : index
  %tile = index.assume %tile_id [lt(%tile_id, %max_tiles)] : index
  %out_group_id = kernel.workgroup.id<y> : index
  %out_group = index.assume %out_group_id [range(%out_group_id, 0, {n // 64 - 1})] : index
  %info_view = buffer.view %tile_info[%origin] : buffer -> view<[%max_tiles]x2xi32>
  %expert32 = view.load %info_view[%tile, %c0] : view<[%max_tiles]x2xi32> -> i32
  %live = scalar.cmpi sge, %expert32, %c0_i32 : i32
  scf.if %live {{
  %expert_bound = scalar.assume %expert32 [range(%expert32, 0, 255)] : i32
  %expert = index.cast %expert_bound : i32 to index
  %expert_origin = index.scale %expert, %expert_bytes : index, offset -> offset
  %workitem_id = kernel.workitem.id<x> : index
  %workitem = index.assume %workitem_id [range(%workitem_id, 0, 255)] : index
  %subgroup = kernel.subgroup.id : index
  %bounded_subgroup = index.assume %subgroup [range(%subgroup, 0, 7)] : index
  %lane = kernel.subgroup.lane.id : index
  %lane_row = index.rem %lane, %c16 : index
  %wave_row_group = index.div %bounded_subgroup, %c4 : index
  %wave_col_group = index.rem %bounded_subgroup, %c4 : index
  %wave_row_value = index.mul %wave_row_group, %c16 : index
  %wave_row = index.assume %wave_row_value [range(%wave_row_value, 0, 16)] : index
  %wave_col_value = index.mul %wave_col_group, %c16 : index
  %wave_col = index.assume %wave_col_value [range(%wave_col_value, 0, 48)] : index
  %w_frag_row = index.add %wave_col, %lane_row : index
  %out_origin = index.mul %out_group, %c64 : index
  %padded_rows = index.mul %max_tiles, %c32 : index
  %rowmap_view = buffer.view %rowmap[%origin] : buffer -> view<[%padded_rows]xi32>
  %stage_bytes = index.constant 9216 : offset
  %a_bytes = index.constant 4608 : offset
  %a_stage = buffer.alloca<workgroup> align(16) %a_bytes : buffer
  %a_stage_view = buffer.view %a_stage[%origin] : buffer -> view<32x72xf16>''')
    for st in stages:
        L.append(f'''  %{st} = buffer.alloca<workgroup> align(16) %stage_bytes : buffer
  %{st}_view = buffer.view %{st}[%origin] : buffer -> view<64x72xf16>''')
    # A staging: thread t loads 8 values of row t/8 at column (t%8)*8.
    if gate_up:
        a_load = '''    %a_assign = view.load %rowmap_view[%a_row_global] : view<[%padded_rows]xi32> -> i32
    %a_valid = scalar.cmpi sge, %a_assign, %c0_i32 : i32
    %a_values = scf.if %a_valid -> (vector<8xf32>) {
      %assign_bound = scalar.assume %a_assign [range(%a_assign, 0, 49151)] : i32
      %assign = index.cast %assign_bound : i32 to index
      %token_value = index.div %assign, %c6 : index
      %token = index.assume %token_value [lt(%token_value, %tokens)] : index
      %v = vector.load %a_view[%token, %a_col] : view<[%tokens]x2048xf32> -> vector<8xf32>
      %sc = view.load %s_view[%token, %c0] : view<[%tokens]x2xf32> -> f32
      %scv = vector.splat %sc : vector<8xf32>
      %vs = vector.mulf %v, %scv : vector<8xf32>
      scf.yield %vs : vector<8xf32>
    } else {
      %zv = vector.constant 0.0 : vector<8xf32>
      scf.yield %zv : vector<8xf32>
    }'''
        a_view = "  %a_view = buffer.view %a[%origin] : buffer -> view<[%tokens]x2048xf32>\n  %s_view = buffer.view %scales[%origin] : buffer -> view<[%tokens]x2xf32>"
    else:
        a_load = '''    %a_raw = vector.load %a_view[%a_row_global, %a_col] : view<[%padded_rows]x512xf32> -> vector<8xf32>
    %sc = view.load %s_view[%a_row_global, %c0] : view<[%padded_rows]x2xf32> -> f32
    %scv = vector.splat %sc : vector<8xf32>
    %a_values = vector.mulf %a_raw, %scv : vector<8xf32>'''
        a_view = "  %a_view = buffer.view %a[%origin] : buffer -> view<[%padded_rows]x512xf32>\n  %s_view = buffer.view %scales[%origin] : buffer -> view<[%padded_rows]x2xf32>"
    L.append(a_view)
    accs = ["g", "u"] if gate_up else ["acc"]
    L.append(f'''  %a_row_value = index.div %workitem, %c8 : index
  %a_row = index.assume %a_row_value [range(%a_row_value, 0, 31)] : index
  %a_quarter = index.rem %workitem, %c8 : index
  %a_stage_col = index.mul %a_quarter, %c8 : index
  %tile_row0 = index.mul %tile, %c32 : index
  %a_row_global_value = index.add %tile_row0, %a_row : index
  %a_row_global = index.assume %a_row_global_value [lt(%a_row_global_value, %padded_rows)] : index
  %stage_row = index.div %workitem, %c4 : index
  %stage_quarter = index.rem %workitem, %c4 : index
  %stage_col = index.mul %stage_quarter, %c16 : index
  %w_out_row = index.add %out_origin, %stage_row : index
  %w_row_add = index.scale %w_out_row, %w_row_bytes : index, offset -> offset
  %w_row_origin = index.add %expert_origin, %w_row_add : offset
  %zero8 = vector.constant 0.0 : vector<8xf32>
  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>
  %k_steps = index.constant {k // 64} : index
  {", ".join("%" + a for a in accs)} = scf.for %step = [%c0 to %k_steps step %c1]({", ".join(f"%{a}_in = %init : vector<8xf32>" for a in accs)}) -> ({", ".join("vector<8xf32>" for _ in accs)}) {{
    %k_origin = index.mul %step, %c64 : index
    %a_col = index.add %k_origin, %a_stage_col : index
{a_load}
    %a_half = vector.fptrunc %a_values : vector<8xf32> to vector<8xf16>
    vector.store %a_half, %a_stage_view[%a_row, %a_stage_col] : vector<8xf16>, view<32x72xf16>
    %w_col = index.add %k_origin, %stage_col : index''')
    for i, (st, w) in enumerate(zip(stages, ["w", "w2"])):
        L.append(dequant16("q4_k", w, None, f"deq{i}").replace("%c256", "%c256").replace("%c2 :", "%c2 :"))
        L.append(f"    vector.store %deq{i}, %{st}_view[%stage_row, %stage_col] : vector<16xf16>, view<64x72xf16>")
    L.append("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    cur = {a: f"%{a}_in" for a in accs}
    for kk in range(4):
        L.append(f'''    %fc{kk} = index.constant {kk * 16} : index
    %lhs{kk} = vector.fragment.load<lhs> %a_stage_view[%wave_row, %fc{kk}] shape [%c16, %c16] : view<32x72xf16> -> vector<16xf16>''')
        for si, (st, a) in enumerate(zip(stages, accs)):
            L.append(f'''    %rv{si}{kk} = vector.load %{st}_view[%w_frag_row, %fc{kk}] : view<64x72xf16> -> vector<16xf16>
    %rhs{si}{kk} = vector.fragment<rhs> %rv{si}{kk} shape [%c16, %c16] : vector<16xf16>
    %m{si}{kk} = vector.mma %lhs{kk}, %rhs{si}{kk}, {cur[a]} : vector<16xf16>, vector<16xf16>, vector<8xf32>''')
            cur[a] = f"%m{si}{kk}"
    L.append("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    L.append(f"    scf.yield {', '.join(cur[a] for a in accs)} : {', '.join('vector<8xf32>' for _ in accs)}\n  }}")
    if gate_up:
        inv_load = '''    %ia = view.load %rowmap_view[%ir] : view<[%padded_rows]xi32> -> i32
    %iv = scalar.cmpi sge, %ia, %c0_i32 : i32
    %inverse = scf.if %iv -> (f32) {
      %ib = scalar.assume %ia [range(%ia, 0, 49151)] : i32
      %ii = index.cast %ib : i32 to index
      %it_value = index.div %ii, %c6 : index
      %it = index.assume %it_value [lt(%it_value, %tokens)] : index
      %v = view.load %s_view[%it, %c1] : view<[%tokens]x2xf32> -> f32
      scf.yield %v : f32
    } else {
      %one = scalar.constant 1.0 : f32
      scf.yield %one : f32
    }'''
    else:
        inv_load = "    %inverse = view.load %s_view[%ir, %c1] : view<[%padded_rows]x2xf32> -> f32"
    L.append(f'''  %inv_bytes = index.constant 2048 : offset
  %inv_tile = buffer.alloca<workgroup> align(16) %inv_bytes : buffer
  %inv_view = buffer.view %inv_tile[%origin] : buffer -> view<32x16xf32>
  %writer = index.cmp ult, %workitem, %c32 : index
  scf.if %writer {{
    %bw = index.assume %workitem [range(%workitem, 0, 31)] : index
    %ir_value = index.add %tile_row0, %bw : index
    %ir = index.assume %ir_value [lt(%ir_value, %padded_rows)] : index
{inv_load}
    %inverse16 = vector.splat %inverse : vector<16xf32>
    vector.store %inverse16, %inv_view[%bw, %c0] : vector<16xf32>, view<32x16xf32>
  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %inv_frag = vector.fragment.load<result> %inv_view[%wave_row, %c0] shape [%c16, %c16] : view<32x16xf32> -> vector<8xf32>''')
    L.append(f'''  %y_layout = encoding.layout.strided [%n, %c1] : encoding<layout>
  %tile_first = index.madd %tile_row0, %n, %out_origin : index
  %tile_offset = index.scale %tile_first, %four : index, offset -> offset
  %y_view = buffer.view %y[%tile_offset] : buffer -> view<32x64xf32, %y_layout>''')
    if gate_up:
        L.append('''  %gs = vector.mulf %g, %inv_frag : vector<8xf32>
  %us = vector.mulf %u, %inv_frag : vector<8xf32>
  %sg = vector.siluf<afn> %gs : vector<8xf32>
  %result = vector.mulf %sg, %us : vector<8xf32>''')
    else:
        L.append("  %result = vector.mulf %acc, %inv_frag : vector<8xf32>")
    L.append('''  vector.fragment.store<result> %result, %y_view[%wave_row, %wave_col] shape [%c16, %c16] : vector<8xf32>, view<32x64xf32, %y_layout>
  }
  kernel.return
}
''')
    return "\n".join(L)





def down_wide():
    import os
    fmt = os.environ.get("FLINT_MOE_FMT", "q4_k")
    from gen_gemm import FMT as _F
    _f = _F[fmt]
    """Expert down projection: A = act_sorted tile (32 x 512) staged once as F16;
    each workgroup produces 32 rows x 128 outputs over 8 K-steps of 64."""
    L = []
    L.append('''kernel.def export("flint_moe_down_grouped") @flint_moe_down_grouped(%max_tiles: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %out_tiles = index.constant 16 : index
  kernel.launch.config workgroups(%max_tiles, %out_tiles, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%max_tiles_arg: index, %tokens_arg: index, %a: buffer, %scales: buffer, %tile_info: buffer, %rowmap: buffer, %w: buffer, %y: buffer) {
  %max_tiles = index.assume %max_tiles_arg [range(%max_tiles_arg, 1, 65536)] : index
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %origin = index.constant 0 : offset
  %sixteen = index.constant 16 : offset
  %four = index.constant 4 : offset
  %block_bytes = index.constant {BLOCK} : offset
  %expert_bytes = index.constant {EXPERT} : offset
  %w_row_bytes = index.constant {ROW} : offset
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
  %n = index.constant 2048 : index
  %c0_i32 = scalar.constant 0 : i32
  %nibble_mask = vector.constant 252645135 : vector<4xi32>
  %tile_id = kernel.workgroup.id<x> : index
  %tile = index.assume %tile_id [lt(%tile_id, %max_tiles)] : index
  %out_group_id = kernel.workgroup.id<y> : index
  %out_group = index.assume %out_group_id [range(%out_group_id, 0, 15)] : index
  %info_view = buffer.view %tile_info[%origin] : buffer -> view<[%max_tiles]x2xi32>
  %expert32 = view.load %info_view[%tile, %c0] : view<[%max_tiles]x2xi32> -> i32
  %live = scalar.cmpi sge, %expert32, %c0_i32 : i32
  scf.if %live {
  %expert_bound = scalar.assume %expert32 [range(%expert32, 0, 255)] : i32
  %expert = index.cast %expert_bound : i32 to index
  %expert_origin = index.scale %expert, %expert_bytes : index, offset -> offset
  %workitem_id = kernel.workitem.id<x> : index
  %workitem = index.assume %workitem_id [range(%workitem_id, 0, 255)] : index
  %subgroup = kernel.subgroup.id : index
  %bounded_subgroup = index.assume %subgroup [range(%subgroup, 0, 7)] : index
  %lane = kernel.subgroup.lane.id : index
  %lane_row = index.rem %lane, %c16 : index
  %wave_row_group = index.div %bounded_subgroup, %c4 : index
  %wave_col_group = index.rem %bounded_subgroup, %c4 : index
  %wave_row_value = index.mul %wave_row_group, %c16 : index
  %wave_row = index.assume %wave_row_value [range(%wave_row_value, 0, 16)] : index
  %wave_col_value = index.mul %wave_col_group, %c32 : index
  %wave_col = index.assume %wave_col_value [range(%wave_col_value, 0, 96)] : index
  %w_frag_row0 = index.add %wave_col, %lane_row : index
  %w_frag_row1 = index.add %w_frag_row0, %c16 : index
  %out_origin = index.mul %out_group, %c128 : index
  %padded_rows = index.mul %max_tiles, %c32 : index
  %tile_row0 = index.mul %tile, %c32 : index
  %a_view = buffer.view %a[%origin] : buffer -> view<[%padded_rows]x512xf32>
  %s_view = buffer.view %scales[%origin] : buffer -> view<[%padded_rows]x2xf32>
  %a_bytes = index.constant 33280 : offset
  %a_stage = buffer.alloca<workgroup> align(16) %a_bytes : buffer
  %a_stage_view = buffer.view %a_stage[%origin] : buffer -> view<32x520xf16>
  %w_bytes = index.constant 18432 : offset
  %w_stage = buffer.alloca<workgroup> align(16) %w_bytes : buffer
  %w_stage_view = buffer.view %w_stage[%origin] : buffer -> view<128x72xf16>
  // Stage the whole A tile: thread t takes row t/8, columns (t%8)*64.
  %a_row_value = index.div %workitem, %c8 : index
  %a_row = index.assume %a_row_value [range(%a_row_value, 0, 31)] : index
  %a_seg = index.rem %workitem, %c8 : index
  %a_col = index.mul %a_seg, %c64 : index
  %a_row_global_value = index.add %tile_row0, %a_row : index
  %a_row_global = index.assume %a_row_global_value [lt(%a_row_global_value, %padded_rows)] : index
  %a_raw = vector.load %a_view[%a_row_global, %a_col] : view<[%padded_rows]x512xf32> -> vector<64xf32>
  %sc = view.load %s_view[%a_row_global, %c0] : view<[%padded_rows]x2xf32> -> f32
  %scv = vector.splat %sc : vector<64xf32>
  %a_scaled = vector.mulf %a_raw, %scv : vector<64xf32>
  %a_half = vector.fptrunc %a_scaled : vector<64xf32> to vector<64xf16>
  vector.store %a_half, %a_stage_view[%a_row, %a_col] : vector<64xf16>, view<32x520xf16>
  // W staging: thread t dequantizes output row t/2, 32 columns at (t%2)*32.
  %stage_row = index.div %workitem, %c2 : index
  %stage_half = index.rem %workitem, %c2 : index
  %stage_col = index.mul %stage_half, %c32 : index
  %stage_col_b = index.add %stage_col, %c16 : index
  %w_out_row = index.add %out_origin, %stage_row : index
  %w_row_add = index.scale %w_out_row, %w_row_bytes : index, offset -> offset
  %w_row_origin = index.add %expert_origin, %w_row_add : offset
  %zero8 = vector.constant 0.0 : vector<8xf32>
  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>
  %k_steps = index.constant 8 : index
  %acc0, %acc1 = scf.for %step = [%c0 to %k_steps step %c1](%x0 = %init : vector<8xf32>, %x1 = %init : vector<8xf32>) -> (vector<8xf32>, vector<8xf32>) {
    %k_origin = index.mul %step, %c64 : index
    %w_col = index.add %k_origin, %stage_col : index''')
    from gen_gemm import dequant16
    L[0] = L[0].replace("{BLOCK}", str(_f["block_bytes"])).replace("{ROW}", str(512 // _f["block_values"] * _f["block_bytes"])).replace("{EXPERT}", str(512 // _f["block_values"] * _f["block_bytes"] * 2048))
    L.append(dequant16(fmt, "w", None, "deqa"))
    L.append("    vector.store %deqa, %w_stage_view[%stage_row, %stage_col] : vector<16xf16>, view<128x72xf16>")
    L.append("    %w_col_b = index.add %k_origin, %stage_col_b : index")
    L.append(dequant16(fmt, "w", None, "deqb").replace("%w_col,", "%w_col_b,").replace("%w_col :", "%w_col_b :"))
    L.append("    vector.store %deqb, %w_stage_view[%stage_row, %stage_col_b] : vector<16xf16>, view<128x72xf16>")
    L.append("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    cur0, cur1 = "%x0", "%x1"
    for kk in range(4):
        L.append(f'''    %fk{kk} = index.constant {kk * 16} : index
    %ak{kk} = index.add %k_origin, %fk{kk} : index
    %lhs{kk} = vector.fragment.load<lhs> %a_stage_view[%wave_row, %ak{kk}] shape [%c16, %c16] : view<32x520xf16> -> vector<16xf16>
    %ra{kk} = vector.load %w_stage_view[%w_frag_row0, %fk{kk}] : view<128x72xf16> -> vector<16xf16>
    %rb{kk} = vector.load %w_stage_view[%w_frag_row1, %fk{kk}] : view<128x72xf16> -> vector<16xf16>
    %fa{kk} = vector.fragment<rhs> %ra{kk} shape [%c16, %c16] : vector<16xf16>
    %fb{kk} = vector.fragment<rhs> %rb{kk} shape [%c16, %c16] : vector<16xf16>
    %m0_{kk} = vector.mma %lhs{kk}, %fa{kk}, {cur0} : vector<16xf16>, vector<16xf16>, vector<8xf32>
    %m1_{kk} = vector.mma %lhs{kk}, %fb{kk}, {cur1} : vector<16xf16>, vector<16xf16>, vector<8xf32>''')
        cur0, cur1 = f"%m0_{kk}", f"%m1_{kk}"
    L.append("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    L.append(f"    scf.yield {cur0}, {cur1} : vector<8xf32>, vector<8xf32>\n  }}")
    L.append('''  %inv_bytes = index.constant 2048 : offset
  %inv_tile = buffer.alloca<workgroup> align(16) %inv_bytes : buffer
  %inv_view = buffer.view %inv_tile[%origin] : buffer -> view<32x16xf32>
  %writer = index.cmp ult, %workitem, %c32 : index
  scf.if %writer {
    %bw = index.assume %workitem [range(%workitem, 0, 31)] : index
    %ir_value = index.add %tile_row0, %bw : index
    %ir = index.assume %ir_value [lt(%ir_value, %padded_rows)] : index
    %inverse = view.load %s_view[%ir, %c1] : view<[%padded_rows]x2xf32> -> f32
    %inverse16 = vector.splat %inverse : vector<16xf32>
    vector.store %inverse16, %inv_view[%bw, %c0] : vector<16xf32>, view<32x16xf32>
  }
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %inv_frag = vector.fragment.load<result> %inv_view[%wave_row, %c0] shape [%c16, %c16] : view<32x16xf32> -> vector<8xf32>
  %r0 = vector.mulf %acc0, %inv_frag : vector<8xf32>
  %r1 = vector.mulf %acc1, %inv_frag : vector<8xf32>
  %y_layout = encoding.layout.strided [%n, %c1] : encoding<layout>
  %tile_first = index.madd %tile_row0, %n, %out_origin : index
  %tile_offset = index.scale %tile_first, %four : index, offset -> offset
  %y_view = buffer.view %y[%tile_offset] : buffer -> view<32x128xf32, %y_layout>
  %wave_col1 = index.add %wave_col, %c16 : index
  vector.fragment.store<result> %r0, %y_view[%wave_row, %wave_col] shape [%c16, %c16] : vector<8xf32>, view<32x128xf32, %y_layout>
  vector.fragment.store<result> %r1, %y_view[%wave_row, %wave_col1] shape [%c16, %c16] : vector<8xf32>, view<32x128xf32, %y_layout>
  }
  kernel.return
}
''')
    return "\n".join(L)


def gate_up_full(fmt="q4_k", chunks=4):
    """Expert gate/up with one workgroup per 32-row tile computing all 512 gate
    and 512 up outputs: A (32 x 64) is staged once per K-step and weights stream
    through workgroup memory in 8 chunks of 64 gate + 64 up rows. Waves: 2 row
    groups x 4 column groups (16 columns of each chunk); 16 fragments per wave."""
    from gen_gemm import FMT
    f = FMT[fmt]
    row_bytes = (2048 // f["block_values"]) * f["block_bytes"]
    L = []
    L.append(f'''kernel.def export("flint_moe_gate_up_grouped") @flint_moe_gate_up_grouped(%max_tiles: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %halves = index.constant {8 // chunks} : index
  kernel.launch.config workgroups(%max_tiles, %halves, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%max_tiles_arg: index, %tokens_arg: index, %a: buffer, %scales: buffer, %tile_info: buffer, %rowmap: buffer, %w: buffer, %w2: buffer, %y: buffer) {{
  %max_tiles = index.assume %max_tiles_arg [range(%max_tiles_arg, 1, 65536)] : index
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %origin = index.constant 0 : offset
  %two = index.constant 2 : offset
  %sixteen = index.constant 16 : offset
  %four = index.constant 4 : offset
  %block_bytes = index.constant {f["block_bytes"]} : offset
  %expert_bytes = index.constant {row_bytes * 512} : offset
  %w_row_bytes = index.constant {row_bytes} : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c6 = index.constant 6 : index
  %c8 = index.constant 8 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %c64 = index.constant 64 : index
  %c256 = index.constant 256 : index
  %n = index.constant 512 : index
  %c0_i32 = scalar.constant 0 : i32
  %nibble_mask = vector.constant 252645135 : vector<4xi32>
  %tile_id = kernel.workgroup.id<x> : index
  %tile = index.assume %tile_id [lt(%tile_id, %max_tiles)] : index
  %part_id = kernel.workgroup.id<y> : index
  %part = index.assume %part_id [range(%part_id, 0, {8 // chunks - 1})] : index
  %part_rows = index.constant {chunks * 64} : index
  %part_origin = index.mul %part, %part_rows : index
  %info_view = buffer.view %tile_info[%origin] : buffer -> view<[%max_tiles]x2xi32>
  %expert32 = view.load %info_view[%tile, %c0] : view<[%max_tiles]x2xi32> -> i32
  %live = scalar.cmpi sge, %expert32, %c0_i32 : i32
  scf.if %live {{
  %expert_bound = scalar.assume %expert32 [range(%expert32, 0, 255)] : i32
  %expert = index.cast %expert_bound : i32 to index
  %expert_origin = index.scale %expert, %expert_bytes : index, offset -> offset
  %workitem_id = kernel.workitem.id<x> : index
  %workitem = index.assume %workitem_id [range(%workitem_id, 0, 255)] : index
  %subgroup = kernel.subgroup.id : index
  %bounded_subgroup = index.assume %subgroup [range(%subgroup, 0, 7)] : index
  %lane = kernel.subgroup.lane.id : index
  %lane_row = index.rem %lane, %c16 : index
  %wave_row_group = index.div %bounded_subgroup, %c4 : index
  %wave_col_group = index.rem %bounded_subgroup, %c4 : index
  %wave_row_value = index.mul %wave_row_group, %c16 : index
  %wave_row = index.assume %wave_row_value [range(%wave_row_value, 0, 16)] : index
  %wave_col_value = index.mul %wave_col_group, %c16 : index
  %wave_col = index.assume %wave_col_value [range(%wave_col_value, 0, 48)] : index
  %w_frag_row = index.add %wave_col, %lane_row : index
  %padded_rows = index.mul %max_tiles, %c32 : index
  %tile_row0 = index.mul %tile, %c32 : index
  %rowmap_view = buffer.view %rowmap[%origin] : buffer -> view<[%padded_rows]xi32>
  %a_view = buffer.view %a[%origin] : buffer -> view<[%tokens]x2048xf16>
  %s_view = buffer.view %scales[%origin] : buffer -> view<[%tokens]x2xf32>
  %a_bytes = index.constant 4608 : offset
  %a_stage = buffer.alloca<workgroup> align(16) %a_bytes : buffer
  %a_stage_view = buffer.view %a_stage[%origin] : buffer -> view<32x72xf16>
  %stage_bytes = index.constant 9216 : offset
  %w_stage = buffer.alloca<workgroup> align(16) %stage_bytes : buffer
  %w_stage_view = buffer.view %w_stage[%origin] : buffer -> view<64x72xf16>
  %w2_stage = buffer.alloca<workgroup> align(16) %stage_bytes : buffer
  %w2_stage_view = buffer.view %w2_stage[%origin] : buffer -> view<64x72xf16>
  // A staging: thread t loads 8 values of row t/8 at column (t%8)*8, gathered by token.
  %a_row_value = index.div %workitem, %c8 : index
  %a_row = index.assume %a_row_value [range(%a_row_value, 0, 31)] : index
  %a_seg = index.rem %workitem, %c8 : index
  %a_stage_col = index.mul %a_seg, %c8 : index
  %a_row_global_value = index.add %tile_row0, %a_row : index
  %a_row_global = index.assume %a_row_global_value [lt(%a_row_global_value, %padded_rows)] : index
  %a_assign = view.load %rowmap_view[%a_row_global] : view<[%padded_rows]xi32> -> i32
  %a_valid = scalar.cmpi sge, %a_assign, %c0_i32 : i32
  %assign_clamped = scalar.maxsi %a_assign, %c0_i32 : i32
  %assign_bound = scalar.assume %assign_clamped [range(%assign_clamped, 0, 49151)] : i32
  %assign = index.cast %assign_bound : i32 to index
  %a_token_value = index.div %assign, %c6 : index
  %a_token = index.assume %a_token_value [lt(%a_token_value, %tokens)] : index
  %zero_f = scalar.constant 0.0 : f32
  %one_h = scalar.constant 1.0 : f16
  %zero_h = scalar.constant 0.0 : f16
  %a_mask_s = scf.select %a_valid, %one_h, %zero_h : f16
  %a_mask = vector.splat %a_mask_s : vector<8xf16>
  // W staging: thread t dequantizes 16 values of chunk row t/4.
  %stage_row = index.div %workitem, %c4 : index
  %stage_quarter = index.rem %workitem, %c4 : index
  %stage_col = index.mul %stage_quarter, %c16 : index
  %zero8 = vector.constant 0.0 : vector<8xf32>
  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>
  %k_steps = index.constant 32 : index''')
    accs = [f"{m}{c}" for c in range(chunks) for m in "gu"]
    L.append("  " + ", ".join(f"%r{a}" for a in accs) + " = scf.for %step = [%c0 to %k_steps step %c1](" +
             ", ".join(f"%x{a} = %init : vector<8xf32>" for a in accs) + ") -> (" + ", ".join("vector<8xf32>" for _ in accs) + ") {")
    L.append('''    %k_origin = index.mul %step, %c64 : index
    %a_col = index.add %k_origin, %a_stage_col : index
    %a_raw = vector.load %a_view[%a_token, %a_col] : view<[%tokens]x2048xf16> -> vector<8xf16>
    %a_half = vector.mulf %a_raw, %a_mask : vector<8xf16>
    vector.store %a_half, %a_stage_view[%a_row, %a_stage_col] : vector<8xf16>, view<32x72xf16>
    %w_col = index.add %k_origin, %stage_col : index''')
    cur = {a: f"%x{a}" for a in accs}
    for c in range(chunks):
        L.append(f'''    %chunk_row{c}_c = index.constant {c * 64} : index
    %chunk_row{c}_p = index.add %chunk_row{c}_c, %part_origin : index
    %chunk_row{c} = index.add %chunk_row{c}_p, %stage_row : index
    %chunk_add{c} = index.scale %chunk_row{c}, %w_row_bytes : index, offset -> offset
    %w_row_origin{c} = index.add %expert_origin, %chunk_add{c} : offset''')
        for wi, w in enumerate(["w", "w2"]):
            L.append(dequant16(fmt, w, None, f"dq{c}{wi}").replace("%w_row_origin", f"%w_row_origin{c}"))
        L.append(f'''    vector.store %dq{c}0, %w_stage_view[%stage_row, %stage_col] : vector<16xf16>, view<64x72xf16>
    vector.store %dq{c}1, %w2_stage_view[%stage_row, %stage_col] : vector<16xf16>, view<64x72xf16>
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)''')
        for kk in range(4):
            L.append(f'''    %fk{c}_{kk} = index.constant {kk * 16} : index
    %lhs{c}_{kk} = vector.fragment.load<lhs> %a_stage_view[%wave_row, %fk{c}_{kk}] shape [%c16, %c16] : view<32x72xf16> -> vector<16xf16>
    %gv{c}_{kk} = vector.load %w_stage_view[%w_frag_row, %fk{c}_{kk}] : view<64x72xf16> -> vector<16xf16>
    %uv{c}_{kk} = vector.load %w2_stage_view[%w_frag_row, %fk{c}_{kk}] : view<64x72xf16> -> vector<16xf16>
    %gf{c}_{kk} = vector.fragment<rhs> %gv{c}_{kk} shape [%c16, %c16] : vector<16xf16>
    %uf{c}_{kk} = vector.fragment<rhs> %uv{c}_{kk} shape [%c16, %c16] : vector<16xf16>
    %mg{c}_{kk} = vector.mma %lhs{c}_{kk}, %gf{c}_{kk}, {cur["g" + str(c)]} : vector<16xf16>, vector<16xf16>, vector<8xf32>
    %mu{c}_{kk} = vector.mma %lhs{c}_{kk}, %uf{c}_{kk}, {cur["u" + str(c)]} : vector<16xf16>, vector<16xf16>, vector<8xf32>''')
            cur["g" + str(c)] = f"%mg{c}_{kk}"; cur["u" + str(c)] = f"%mu{c}_{kk}"
        L.append("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    L.append("    scf.yield " + ", ".join(cur[a] for a in accs) + " : " + ", ".join("vector<8xf32>" for _ in accs) + "\n  }")
    L.append('''  %inv_bytes = index.constant 2048 : offset
  %inv_tile = buffer.alloca<workgroup> align(16) %inv_bytes : buffer
  %inv_view = buffer.view %inv_tile[%origin] : buffer -> view<32x16xf32>
  %writer = index.cmp ult, %workitem, %c32 : index
  scf.if %writer {
    %bw = index.assume %workitem [range(%workitem, 0, 31)] : index
    %ir_value = index.add %tile_row0, %bw : index
    %ir = index.assume %ir_value [lt(%ir_value, %padded_rows)] : index
    %ia = view.load %rowmap_view[%ir] : view<[%padded_rows]xi32> -> i32
    %iv = scalar.cmpi sge, %ia, %c0_i32 : i32
    %ic = scalar.maxsi %ia, %c0_i32 : i32
    %ib = scalar.assume %ic [range(%ic, 0, 49151)] : i32
    %ii = index.cast %ib : i32 to index
    %it_value = index.div %ii, %c6 : index
    %it = index.assume %it_value [lt(%it_value, %tokens)] : index
    %iraw = view.load %s_view[%it, %c1] : view<[%tokens]x2xf32> -> f32
    %one = scalar.constant 1.0 : f32
    %inverse = scf.select %iv, %iraw, %one : f32
    %inverse16 = vector.splat %inverse : vector<16xf32>
    vector.store %inverse16, %inv_view[%bw, %c0] : vector<16xf32>, view<32x16xf32>
  }
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %inv_frag = vector.fragment.load<result> %inv_view[%wave_row, %c0] shape [%c16, %c16] : view<32x16xf32> -> vector<8xf32>
  %y_layout = encoding.layout.strided [%n, %c1] : encoding<layout>
  %tile_first = index.mul %tile_row0, %n : index
  %tile_offset = index.scale %tile_first, %four : index, offset -> offset
  %y_view = buffer.view %y[%tile_offset] : buffer -> view<32x512xf32, %y_layout>''')
    for c in range(chunks):
        L.append(f'''  %gs{c} = vector.mulf %rg{c}, %inv_frag : vector<8xf32>
  %us{c} = vector.mulf %ru{c}, %inv_frag : vector<8xf32>
  %sg{c} = vector.siluf<afn> %gs{c} : vector<8xf32>
  %out{c} = vector.mulf %sg{c}, %us{c} : vector<8xf32>
  %oc{c}_c = index.constant {c * 64} : index
  %oc{c}_p = index.add %oc{c}_c, %part_origin : index
  %oc{c} = index.add %oc{c}_p, %wave_col : index
  vector.fragment.store<result> %out{c}, %y_view[%wave_row, %oc{c}] shape [%c16, %c16] : vector<8xf32>, view<32x512xf32, %y_layout>''')
    L.append("  }\n  kernel.return\n}\n")
    return "\n".join(L)


def down_full(fmt="q4_k", groups=4):
    """Expert down: A (32 x 512) staged once per workgroup; each workgroup computes
    `groups` of the 16 groups of 128 outputs (8 K-steps each)."""
    src = down_wide()
    src = src.replace('''  %out_tiles = index.constant 16 : index''', f'''  %out_tiles = index.constant {16 // groups} : index''')
    src = src.replace('''  %out_group_id = kernel.workgroup.id<y> : index
  %out_group = index.assume %out_group_id [range(%out_group_id, 0, 15)] : index
''', f'''  %out_part_id = kernel.workgroup.id<y> : index
  %out_part = index.assume %out_part_id [range(%out_part_id, 0, {16 // groups - 1})] : index
''')
    src = src.replace("  %out_origin = index.mul %out_group, %c128 : index\n", "")
    inv_alloc = '''  %inv_bytes = index.constant 2048 : offset
  %inv_tile = buffer.alloca<workgroup> align(16) %inv_bytes : buffer
'''
    src = src.replace(inv_alloc, "")
    loop_start = inv_alloc + f'''  %group_count = index.constant {groups} : index
  %group_base = index.mul %out_part, %group_count : index
  scf.for %og = [%c0 to %group_count step %c1] {{
  %out_group_value = index.add %group_base, %og : index
  %out_group = index.assume %out_group_value [range(%out_group_value, 0, 15)] : index
  %out_origin = index.mul %out_group, %c128 : index
  // W staging: thread t dequantizes output row t/2, 32 columns at (t%2)*32.'''
    src = src.replace("  // W staging: thread t dequantizes output row t/2, 32 columns at (t%2)*32.", loop_start)
    tail = "  }\n  kernel.return\n}\n"
    assert src.endswith(tail)
    src = src[: -len(tail)] + "  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)\n  }\n" + tail
    return src



def gate_up_r(fmt, R):
    """Expert gate/up over R-row tiles (R in 32/64/128): one workgroup per tile
    and 64-column part computes 64 gate + 64 up outputs. A (R x 64, F16
    pre-scaled) and the dequantized weight chunk are staged per K-step; waves
    are (R/16) row groups x (128/R) column groups of 64*16/R... columns, so each
    staged weight value feeds R rows."""
    from gen_gemm import FMT
    f = FMT[fmt]
    row_bytes = (2048 // f["block_values"]) * f["block_bytes"]
    RG = R // 16; CG = 8 // RG; CF = 64 // CG // 16; P = R // 32
    L = []
    L.append(f"""kernel.def export("flint_moe_gate_up_grouped") @flint_moe_gate_up_grouped(%max_tiles: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %parts = index.constant 8 : index
  kernel.launch.config workgroups(%max_tiles, %parts, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%max_tiles_arg: index, %tokens_arg: index, %a: buffer, %scales: buffer, %tile_info: buffer, %rowmap: buffer, %w: buffer, %w2: buffer, %y: buffer) {{
  %max_tiles = index.assume %max_tiles_arg [range(%max_tiles_arg, 1, 65536)] : index
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %origin = index.constant 0 : offset
  %two = index.constant 2 : offset
  %sixteen = index.constant 16 : offset
  %four = index.constant 4 : offset
  %block_bytes = index.constant {f["block_bytes"]} : offset
  %expert_bytes = index.constant {row_bytes * 512} : offset
  %w_row_bytes = index.constant {row_bytes} : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c6 = index.constant 6 : index
  %c8 = index.constant 8 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %c64 = index.constant 64 : index
  %c256 = index.constant 256 : index
  %cR = index.constant {R} : index
  %cCG = index.constant {CG} : index
  %cWC = index.constant {CF * 16} : index
  %n = index.constant 512 : index
  %c0_i32 = scalar.constant 0 : i32
  %nibble_mask = vector.constant 252645135 : vector<4xi32>
  %tile_id = kernel.workgroup.id<x> : index
  %tile = index.assume %tile_id [lt(%tile_id, %max_tiles)] : index
  %part_id = kernel.workgroup.id<y> : index
  %part = index.assume %part_id [range(%part_id, 0, 7)] : index
  %part_origin = index.mul %part, %c64 : index
  %info_view = buffer.view %tile_info[%origin] : buffer -> view<[%max_tiles]x2xi32>
  %expert32 = view.load %info_view[%tile, %c0] : view<[%max_tiles]x2xi32> -> i32
  %live = scalar.cmpi sge, %expert32, %c0_i32 : i32
  scf.if %live {{
  %expert_bound = scalar.assume %expert32 [range(%expert32, 0, 255)] : i32
  %expert = index.cast %expert_bound : i32 to index
  %expert_origin = index.scale %expert, %expert_bytes : index, offset -> offset
  %workitem_id = kernel.workitem.id<x> : index
  %workitem = index.assume %workitem_id [range(%workitem_id, 0, 255)] : index
  %subgroup = kernel.subgroup.id : index
  %bounded_subgroup = index.assume %subgroup [range(%subgroup, 0, 7)] : index
  %lane = kernel.subgroup.lane.id : index
  %lane_row = index.rem %lane, %c16 : index
  %wave_row_group = index.div %bounded_subgroup, %cCG : index
  %wave_col_group = index.rem %bounded_subgroup, %cCG : index
  %wave_row_value = index.mul %wave_row_group, %c16 : index
  %wave_row = index.assume %wave_row_value [range(%wave_row_value, 0, {R - 16})] : index
  %wave_col_value = index.mul %wave_col_group, %cWC : index
  %wave_col = index.assume %wave_col_value [range(%wave_col_value, 0, {64 - CF * 16})] : index
  %padded_rows = index.mul %max_tiles, %cR : index
  %tile_row0 = index.mul %tile, %cR : index
  %rowmap_view = buffer.view %rowmap[%origin] : buffer -> view<[%padded_rows]xi32>
  %a_view = buffer.view %a[%origin] : buffer -> view<[%tokens]x2048xf16>
  %s_view = buffer.view %scales[%origin] : buffer -> view<[%tokens]x2xf32>
  %a_bytes = index.constant {R * 72 * 2} : offset
  %a_stage = buffer.alloca<workgroup> align(16) %a_bytes : buffer
  %a_stage_view = buffer.view %a_stage[%origin] : buffer -> view<{R}x72xf16>
  %stage_bytes = index.constant 9216 : offset
  %w_stage = buffer.alloca<workgroup> align(16) %stage_bytes : buffer
  %w_stage_view = buffer.view %w_stage[%origin] : buffer -> view<64x72xf16>
  %w2_stage = buffer.alloca<workgroup> align(16) %stage_bytes : buffer
  %w2_stage_view = buffer.view %w2_stage[%origin] : buffer -> view<64x72xf16>
  %a_row_base_value = index.div %workitem, %c8 : index
  %a_seg = index.rem %workitem, %c8 : index
  %a_stage_col = index.mul %a_seg, %c8 : index
  %one_h = scalar.constant 1.0 : f16
  %zero_h = scalar.constant 0.0 : f16""")
    for p_ in range(P):
        L.append(f"""  %a_row{p_}_value = index.add %a_row_base_value, %c{32 * p_ if p_ else 0}_r : index
  %a_row{p_} = index.assume %a_row{p_}_value [range(%a_row{p_}_value, 0, {R - 1})] : index
  %a_rg{p_}_value = index.add %tile_row0, %a_row{p_} : index
  %a_rg{p_} = index.assume %a_rg{p_}_value [lt(%a_rg{p_}_value, %padded_rows)] : index
  %a_as{p_} = view.load %rowmap_view[%a_rg{p_}] : view<[%padded_rows]xi32> -> i32
  %a_ok{p_} = scalar.cmpi sge, %a_as{p_}, %c0_i32 : i32
  %a_cl{p_} = scalar.maxsi %a_as{p_}, %c0_i32 : i32
  %a_bd{p_} = scalar.assume %a_cl{p_} [range(%a_cl{p_}, 0, 49151)] : i32
  %a_ix{p_} = index.cast %a_bd{p_} : i32 to index
  %a_tv{p_} = index.div %a_ix{p_}, %c6 : index
  %a_token{p_} = index.assume %a_tv{p_} [lt(%a_tv{p_}, %tokens)] : index
  %a_ms{p_} = scf.select %a_ok{p_}, %one_h, %zero_h : f16
  %a_mask{p_} = vector.splat %a_ms{p_} : vector<8xf16>""")
    L.insert(-P, "".join(f"  %c{32 * p_}_r = index.constant {32 * p_} : index\n" for p_ in range(P)).rstrip("\n").replace("%c0_r = index.constant 0", "%c0_r = index.constant 0"))
    L.append("""  %stage_row = index.div %workitem, %c4 : index
  %stage_quarter = index.rem %workitem, %c4 : index
  %stage_col = index.mul %stage_quarter, %c16 : index
  %w_row_value = index.add %part_origin, %stage_row : index
  %w_row_add = index.scale %w_row_value, %w_row_bytes : index, offset -> offset
  %w_row_origin = index.add %expert_origin, %w_row_add : offset
  %zero8 = vector.constant 0.0 : vector<8xf32>
  %init = vector.fragment<init> %zero8 shape [%c16, %c16] : vector<8xf32>
  %k_steps = index.constant 32 : index""")
    accs = [f"{m}{c}" for m in "gu" for c in range(CF)]
    L.append("  " + ", ".join(f"%r{a}" for a in accs) + " = scf.for %step = [%c0 to %k_steps step %c1](" +
             ", ".join(f"%x{a} = %init : vector<8xf32>" for a in accs) + ") -> (" + ", ".join("vector<8xf32>" for _ in accs) + ") {")
    L.append("    %k_origin = index.mul %step, %c64 : index\n    %a_col = index.add %k_origin, %a_stage_col : index")
    for p_ in range(P):
        L.append(f"""    %a_raw{p_} = vector.load %a_view[%a_token{p_}, %a_col] : view<[%tokens]x2048xf16> -> vector<8xf16>
    %a_half{p_} = vector.mulf %a_raw{p_}, %a_mask{p_} : vector<8xf16>
    vector.store %a_half{p_}, %a_stage_view[%a_row{p_}, %a_stage_col] : vector<8xf16>, view<{R}x72xf16>""")
    L.append("    %w_col = index.add %k_origin, %stage_col : index")
    for wi, w in enumerate(["w", "w2"]):
        L.append(dequant16(fmt, w, None, f"dq{wi}"))
    L.append("""    vector.store %dq0, %w_stage_view[%stage_row, %stage_col] : vector<16xf16>, view<64x72xf16>
    vector.store %dq1, %w2_stage_view[%stage_row, %stage_col] : vector<16xf16>, view<64x72xf16>
    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)""")
    cur = {a: f"%x{a}" for a in accs}
    for kk in range(4):
        L.append(f"""    %fk{kk} = index.constant {kk * 16} : index
    %lhs{kk} = vector.fragment.load<lhs> %a_stage_view[%wave_row, %fk{kk}] shape [%c16, %c16] : view<{R}x72xf16> -> vector<16xf16>""")
        for c in range(CF):
            L.append(f"""    %fr{c}_{kk}_c = index.constant {c * 16} : index
    %fr{c}_{kk}_a = index.add %wave_col, %fr{c}_{kk}_c : index
    %fr{c}_{kk} = index.add %fr{c}_{kk}_a, %lane_row : index
    %gv{c}_{kk} = vector.load %w_stage_view[%fr{c}_{kk}, %fk{kk}] : view<64x72xf16> -> vector<16xf16>
    %uv{c}_{kk} = vector.load %w2_stage_view[%fr{c}_{kk}, %fk{kk}] : view<64x72xf16> -> vector<16xf16>
    %gf{c}_{kk} = vector.fragment<rhs> %gv{c}_{kk} shape [%c16, %c16] : vector<16xf16>
    %uf{c}_{kk} = vector.fragment<rhs> %uv{c}_{kk} shape [%c16, %c16] : vector<16xf16>
    %mg{c}_{kk} = vector.mma %lhs{kk}, %gf{c}_{kk}, {cur["g" + str(c)]} : vector<16xf16>, vector<16xf16>, vector<8xf32>
    %mu{c}_{kk} = vector.mma %lhs{kk}, %uf{c}_{kk}, {cur["u" + str(c)]} : vector<16xf16>, vector<16xf16>, vector<8xf32>""")
            cur["g" + str(c)] = f"%mg{c}_{kk}"; cur["u" + str(c)] = f"%mu{c}_{kk}"
    L.append("    kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    L.append("    scf.yield " + ", ".join(cur[a] for a in accs) + " : " + ", ".join("vector<8xf32>" for _ in accs) + "\n  }")
    L.append(f"""  %inv_bytes = index.constant {R * 64} : offset
  %inv_tile = buffer.alloca<workgroup> align(16) %inv_bytes : buffer
  %inv_view = buffer.view %inv_tile[%origin] : buffer -> view<{R}x16xf32>
  %writer = index.cmp ult, %workitem, %cR : index
  scf.if %writer {{
    %bw = index.assume %workitem [range(%workitem, 0, {R - 1})] : index
    %ir_value = index.add %tile_row0, %bw : index
    %ir = index.assume %ir_value [lt(%ir_value, %padded_rows)] : index
    %ia = view.load %rowmap_view[%ir] : view<[%padded_rows]xi32> -> i32
    %iv = scalar.cmpi sge, %ia, %c0_i32 : i32
    %ic = scalar.maxsi %ia, %c0_i32 : i32
    %ib = scalar.assume %ic [range(%ic, 0, 49151)] : i32
    %ii = index.cast %ib : i32 to index
    %it_value = index.div %ii, %c6 : index
    %it = index.assume %it_value [lt(%it_value, %tokens)] : index
    %iraw = view.load %s_view[%it, %c1] : view<[%tokens]x2xf32> -> f32
    %one = scalar.constant 1.0 : f32
    %inverse = scf.select %iv, %iraw, %one : f32
    %inverse16 = vector.splat %inverse : vector<16xf32>
    vector.store %inverse16, %inv_view[%bw, %c0] : vector<16xf32>, view<{R}x16xf32>
  }}
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %inv_frag = vector.fragment.load<result> %inv_view[%wave_row, %c0] shape [%c16, %c16] : view<{R}x16xf32> -> vector<8xf32>
  %y_layout = encoding.layout.strided [%n, %c1] : encoding<layout>
  %tile_first = index.mul %tile_row0, %n : index
  %tile_offset = index.scale %tile_first, %four : index, offset -> offset
  %y_view = buffer.view %y[%tile_offset] : buffer -> view<{R}x512xf32, %y_layout>""")
    for c in range(CF):
        L.append(f"""  %gs{c} = vector.mulf %rg{c}, %inv_frag : vector<8xf32>
  %us{c} = vector.mulf %ru{c}, %inv_frag : vector<8xf32>
  %sg{c} = vector.siluf<afn> %gs{c} : vector<8xf32>
  %out{c} = vector.mulf %sg{c}, %us{c} : vector<8xf32>
  %oc{c}_c = index.constant {c * 16} : index
  %oc{c}_p = index.add %oc{c}_c, %part_origin : index
  %oc{c} = index.add %oc{c}_p, %wave_col : index
  vector.fragment.store<result> %out{c}, %y_view[%wave_row, %oc{c}] shape [%c16, %c16] : vector<8xf32>, view<{R}x512xf32, %y_layout>""")
    L.append("  }\n  kernel.return\n}\n")
    return "\n".join(L)


def bucket_r(R):
    import re
    b = BUCKET.replace("into 32-row tiles", f"into {R}-row tiles")
    b = re.sub(r"%c32\b(?!_)", "%cR", b).replace("%cR = index.constant 32 : index", f"%cR = index.constant {R} : index")
    b = b.replace("%c31_i32 = scalar.constant 31 : i32", f"%c31_i32 = scalar.constant {R - 1} : i32")
    b = b.replace("%c5_i32 = scalar.constant 5 : i32", f"%c5_i32 = scalar.constant {R.bit_length() - 1} : i32")
    return b


def down_virtual(R, groups):
    """down_full over R-row bucket tiles: each R-row tile runs as R/32 virtual
    32-row tiles (max_tiles here counts virtual tiles); empty halves exit."""
    src = down_full(groups=groups)
    if R == 32:
        return src
    K = R // 32
    old = """  %info_view = buffer.view %tile_info[%origin] : buffer -> view<[%max_tiles]x2xi32>
  %expert32 = view.load %info_view[%tile, %c0] : view<[%max_tiles]x2xi32> -> i32
  %live = scalar.cmpi sge, %expert32, %c0_i32 : i32"""
    assert old in src
    new = f"""  %vk = index.constant {K} : index
  %real_tiles = index.div %max_tiles, %vk : index
  %real_value = index.div %tile, %vk : index
  %real_tile = index.assume %real_value [lt(%real_value, %real_tiles)] : index
  %info_view = buffer.view %tile_info[%origin] : buffer -> view<[%real_tiles]x2xi32>
  %expert32 = view.load %info_view[%real_tile, %c0] : view<[%real_tiles]x2xi32> -> i32
  %expert_live = scalar.cmpi sge, %expert32, %c0_i32 : i32
  %c32v = index.constant 32 : index
  %vrows = index.mul %max_tiles, %c32v : index
  %vfirst_value = index.mul %tile, %c32v : index
  %vfirst = index.assume %vfirst_value [lt(%vfirst_value, %vrows)] : index
  %vmap = buffer.view %rowmap[%origin] : buffer -> view<[%vrows]xi32>
  %vassign = view.load %vmap[%vfirst] : view<[%vrows]xi32> -> i32
  %rows_live = scalar.cmpi sge, %vassign, %c0_i32 : i32
  %live = scalar.andi %expert_live, %rows_live : i1"""
    return src.replace(old, new)


if __name__ == "__main__":
    import os
    R = int(os.environ.get("MOE_TILE", "32"))
    fmt = os.environ.get("FLINT_MOE_FMT", "q4_k")
    sys.stdout.write("// Generated by gen_moe.py: Flint prefill MoE kernels.\n" + bucket_r(R) + "\n" + gate_up_r(fmt, R) + "\n" + down_virtual(R, int(os.environ.get("DOWN_GROUPS", "8"))))
