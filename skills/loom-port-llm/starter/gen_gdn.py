"""Gated DeltaNet (linear attention) kernels for hybrid models such as Qwen3.5 / Qwen3-Next
(llama.cpp: src/models/qwen35.cpp + delta-net-base.cpp). gen_llm.py includes them when the
config has "recurrent_layers".

Per layer and token: qkv = W_qkv h (C = 2*HK*DK + HV*DV channels), z = W_z h, alpha/beta = W h
(HV each). Causal conv1d (kernel K) over qkv + SiLU; q, k L2-normalized per head (q also scaled
by 1/sqrt(DK)); per value head with state S[a][b] (a: key dim, b: value dim):
    S *= exp(softplus(alpha + dt_bias) * ssm_a)
    d  = sigmoid(beta) * (v - S^T k);  S += k (x) d;  o = S^T q
then RMSNorm(o) * ssm_norm * silu(z) per head and the output projection. Value head h reads
key head h % HK (ggml_repeat tiles the key heads).

Recurrent state lives in the runtime's recurrent slots (storage.loom): the slot holds, per
recurrent layer, S (HV*DV*DK f32, column-major per head: index (h*DV + b)*DK + a) followed
by the conv history (K-1 previous qkv inputs, [K-1][C] f32, oldest first). Per row, the
origins record names a reader slot (state at the fork position, e.g. a reused prefix), a
writer slot (the row's private state) and the fork position: a span starting at the fork
reads the reader slot, otherwise the writer; a span at position 0 starts from zeros. Every
span publishes its final state to the writer. The register recurrence is adapted from
hrx-loom-library model/Qwen/Qwen3.8-27B/program/kernel/gdn_prefill.loom.
"""


def gdn_geometry(cfg):
    g = cfg["gdn"]
    HK, HV, DK, DV, K = g["k_heads"], g["v_heads"], g["head_k"], g["head_v"], g["conv_kernel"]
    C = 2 * HK * DK + HV * DV
    state_bytes = HV * DV * DK * 4
    history_bytes = (K - 1) * C * 4
    layer_bytes = (state_bytes + history_bytes + 255) // 256 * 256
    return dict(HK=HK, HV=HV, DK=DK, DV=DV, K=K, C=C, VD=HV * DV, state_bytes=state_bytes,
                layer_bytes=layer_bytes, slot_bytes=layer_bytes * len(cfg["recurrent_layers"]))


SLOT = r"""
// (reader, writer, fresh) for the span of `row` starting at `start`: the reader slot at the
// fork position, otherwise the row's own writer; position 0 starts from zeros.
func.def inline @g_gdn_slots(%origins: buffer, %row: index, %start32: i32, %layer_bytes: offset) -> (offset, offset, i1) {
  %record_stride = index.constant 40 : offset
  %bounded_row = index.assume %row [range(%row, 0, 2147483646)] : index
  %record_offset = index.scale %bounded_row, %record_stride : index, offset -> offset
  %records = buffer.view %origins[%record_offset] : buffer -> view<5xi64>
  %read64 = view.load %records[0] : view<5xi64> -> i64
  %write64 = view.load %records[1] : view<5xi64> -> i64
  %fork64 = view.load %records[3] : view<5xi64> -> i64
  %start64 = scalar.extui %start32 : i32 to i64
  %at_fork = scalar.cmpi eq, %start64, %fork64 : i64
  %reader64 = scf.select %at_fork, %read64, %write64 : i64
  %reader_bound = scalar.assume %reader64 [range(%reader64, 0, 9223372036854775807)] : i64
  %writer_bound = scalar.assume %write64 [range(%write64, 0, 9223372036854775807)] : i64
  %reader_base = index.cast %reader_bound : i64 to offset
  %writer_base = index.cast %writer_bound : i64 to offset
  %reader = index.add %reader_base, %layer_bytes : offset
  %writer = index.add %writer_base, %layer_bytes : offset
  %zero32 = scalar.constant 0 : i32
  %fresh = scalar.cmpi eq, %start32, %zero32 : i32
  func.return %reader, %writer, %fresh : offset, offset, i1
}

// Span `span` of the epoch: (active, length, start position, row, first input slot).
func.def inline @g_gdn_span(%metadata: buffer, %spans: index, %span: index) -> (i1, index, i32, index, index) {
  %zero = index.constant 0 : offset
  %c16 = index.constant 16 : offset
  %header = buffer.view %metadata[%zero] : buffer -> view<4xi32>
  %count32 = view.load %header[0] : view<4xi32> -> i32
  %count_bound = scalar.assume %count32 [range(%count32, 0, 8192)] : i32
  %count = index.cast %count_bound : i32 to index
  %active = index.cmp ult, %span, %count : index
  %bounded_span = index.assume %span [lt(%span, %spans)] : index
  %packets = buffer.view %metadata[%c16] : buffer -> view<[%spans]x9xi32>
  %length32 = view.load %packets[%bounded_span, 0] : view<[%spans]x9xi32> -> i32
  %start32 = view.load %packets[%bounded_span, 1] : view<[%spans]x9xi32> -> i32
  %row32 = view.load %packets[%bounded_span, 3] : view<[%spans]x9xi32> -> i32
  %first32 = view.load %packets[%bounded_span, 4] : view<[%spans]x9xi32> -> i32
  %length_bound = scalar.assume %length32 [range(%length32, 0, 8192)] : i32
  %row_bound = scalar.assume %row32 [range(%row32, 0, 2147483646)] : i32
  %first_bound = scalar.assume %first32 [range(%first32, 0, 8191)] : i32
  %length = index.cast %length_bound : i32 to index
  %row = index.cast %row_bound : i32 to index
  %first = index.cast %first_bound : i32 to index
  %zero_index = index.constant 0 : index
  %nonempty = index.cmp ugt, %length, %zero_index : index
  %live = scalar.andi %active, %nonempty : i1
  func.return %live, %length, %start32, %row, %first : i1, index, i32, index, index
}
"""


def gdn_kernels(cfg, eps):
    G = gdn_geometry(cfg)
    HK, HV, DK, DV, K, C, VD = G["HK"], G["HV"], G["DK"], G["DV"], G["K"], G["C"], G["VD"]
    NG = len(cfg["recurrent_layers"])
    SB, LB = G["state_bytes"], G["layer_bytes"]
    RK, RV = DK // 32, DV // 32
    assert DK % 32 == 0 and DV % 32 == 0 and DV % 4 == 0 and HV % HK == 0
    T = [SLOT]
    # ---- conv1d + SiLU over the span's inputs and the stored history
    taps = []
    for j in range(K):
        back = K - 1 - j
        taps.append(f"""      %w{j} = view.load %w_view[%ch, {j}] : view<{C}x{K}xf32> -> f32
      %back{j} = index.constant {back} : index
      %has{j} = index.cmp uge, %rel, %back{j} : index
      %x{j} = scf.if %has{j} -> (f32) {{
        %src{j} = index.sub %token, %back{j} : index
        %bsrc{j} = index.assume %src{j} [lt(%src{j}, %tokens)] : index
        %v{j} = view.load %qkv_view[%bsrc{j}, %ch] : view<[%tokens]x{C}xf32> -> f32
        scf.yield %v{j} : f32
      }} else {{
        %h{j} = scf.if %fresh -> (f32) {{
          scf.yield %zero_f : f32
        }} else {{
          %hk{j}a = index.constant {K - 1 - back} : index
          %hk{j}b = index.add %hk{j}a, %rel : index
          %hk{j} = index.assume %hk{j}b [range(%hk{j}b, 0, {K - 2})] : index
          %hv{j} = view.load %hist_view[%hk{j}, %ch] : view<{K - 1}x{C}xf32> -> f32
          scf.yield %hv{j} : f32
        }}
        scf.yield %h{j} : f32
      }}
      %acc{j} = scalar.fmaf %w{j}, %x{j}, {"%zero_f" if j == 0 else f"%acc{j - 1}"} : f32""")
    T.append(f"""
// out[t, c] = silu(sum_j w[c, j] x[t - (K-1) + j, c]); inputs before the span come from the history.
kernel.def export("g_gdn_conv") @g_gdn_conv(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %spans_arg: index, %layer_arg: index, %metadata: buffer, %info: buffer, %origins: buffer, %arena: buffer, %qkv: buffer, %weight: buffer, %out: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %layer = index.assume %layer_arg [range(%layer_arg, 0, {NG - 1})] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %thread_id = kernel.workitem.id<x> : index
  %thread = index.assume %thread_id [range(%thread_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0_i32 = scalar.constant 0 : i32
  %zero_f = scalar.constant 0.0 : f32
  %one_f = scalar.constant 1.0 : f32
  %cC = index.constant {C} : index
  %c256 = index.constant 256 : index
  %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %live32 = view.load %info_view[%token, 2] : view<[%tokens]x4xi32> -> i32
  %span32 = view.load %info_view[%token, 3] : view<[%tokens]x4xi32> -> i32
  %live = scalar.cmpi ne, %live32, %c0_i32 : i32
  scf.if %live {{
    %span_bound = scalar.assume %span32 [range(%span32, 0, 8191)] : i32
    %span = index.cast %span_bound : i32 to index
    %active, %length, %start32, %row, %first = func.call @g_gdn_span(%metadata, %spans, %span) : (buffer, index, index) -> (i1, index, i32, index, index)
    %layer_stride = index.constant {LB} : offset
    %layer_bytes = index.scale %layer, %layer_stride : index, offset -> offset
    %reader, %writer, %fresh = func.call @g_gdn_slots(%origins, %row, %start32, %layer_bytes) : (buffer, index, i32, offset) -> (offset, offset, i1)
    %history_skip = index.constant {SB} : offset
    %history_at = index.add %reader, %history_skip : offset
    %hist_view = buffer.view %arena[%history_at] : buffer -> view<{K - 1}x{C}xf32>
    %qkv_view = buffer.view %qkv[%zero] : buffer -> view<[%tokens]x{C}xf32>
    %out_view = buffer.view %out[%zero] : buffer -> view<[%tokens]x{C}xf32>
    %w_view = buffer.view %weight[%zero] : buffer -> view<{C}x{K}xf32>
    %rel_value = index.sub %token, %first : index
    %rel = index.assume %rel_value [range(%rel_value, 0, 8191)] : index
    scf.for %c = [%thread to %cC step %c256] {{
      %ch = index.assume %c [range(%c, 0, {C - 1})] : index
{chr(10).join(taps)}
      %sig = scalar.logisticf<afn> %acc{K - 1} : f32
      %silu = scalar.mulf %acc{K - 1}, %sig : f32
      view.store %silu, %out_view[%token, %ch] : f32, view<[%tokens]x{C}xf32>
    }}
  }}
  kernel.return
}}
""")
    # ---- history commit: the last K-1 inputs of each span (after g_gdn_conv read the old ones)
    reads, writes = [], []
    for k in range(K - 1):
        reads.append(f"""      %need{k} = index.constant {K - 1 - k} : index
      %has{k} = index.cmp uge, %length, %need{k} : index
      %val{k} = scf.if %has{k} -> (f32) {{
        %off{k}a = index.add %first, %length : index
        %src{k} = index.sub %off{k}a, %need{k} : index
        %bsrc{k} = index.assume %src{k} [lt(%src{k}, %tokens)] : index
        %v{k} = view.load %qkv_view[%bsrc{k}, %ch] : view<[%tokens]x{C}xf32> -> f32
        scf.yield %v{k} : f32
      }} else {{
        %o{k} = scf.if %fresh -> (f32) {{
          scf.yield %zero_f : f32
        }} else {{
          %hk{k}a = index.constant {k} : index
          %hk{k}b = index.add %hk{k}a, %length : index
          %hk{k} = index.assume %hk{k}b [range(%hk{k}b, 0, {K - 2})] : index
          %hv{k} = view.load %old_view[%hk{k}, %ch] : view<{K - 1}x{C}xf32> -> f32
          scf.yield %hv{k} : f32
        }}
        scf.yield %o{k} : f32
      }}""")
        writes.append(f"      view.store %val{k}, %new_view[{k}, %ch] : f32, view<{K - 1}x{C}xf32>")
    T.append(f"""
kernel.def export("g_gdn_conv_commit") @g_gdn_conv_commit(%spans: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%spans, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %spans_arg: index, %layer_arg: index, %metadata: buffer, %origins: buffer, %arena: buffer, %qkv: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %layer = index.assume %layer_arg [range(%layer_arg, 0, {NG - 1})] : index
  %span_id = kernel.workgroup.id<x> : index
  %span = index.assume %span_id [lt(%span_id, %spans)] : index
  %thread_id = kernel.workitem.id<x> : index
  %thread = index.assume %thread_id [range(%thread_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %zero_f = scalar.constant 0.0 : f32
  %cC = index.constant {C} : index
  %c256 = index.constant 256 : index
  %active, %length, %start32, %row, %first = func.call @g_gdn_span(%metadata, %spans, %span) : (buffer, index, index) -> (i1, index, i32, index, index)
  scf.if %active {{
    %layer_stride = index.constant {LB} : offset
    %layer_bytes = index.scale %layer, %layer_stride : index, offset -> offset
    %reader, %writer, %fresh = func.call @g_gdn_slots(%origins, %row, %start32, %layer_bytes) : (buffer, index, i32, offset) -> (offset, offset, i1)
    %history_skip = index.constant {SB} : offset
    %old_at = index.add %reader, %history_skip : offset
    %new_at = index.add %writer, %history_skip : offset
    %old_view = buffer.view %arena[%old_at] : buffer -> view<{K - 1}x{C}xf32>
    %new_view = buffer.view %arena[%new_at] : buffer -> view<{K - 1}x{C}xf32>
    %qkv_view = buffer.view %qkv[%zero] : buffer -> view<[%tokens]x{C}xf32>
    scf.for %c = [%thread to %cC step %c256] {{
      %ch = index.assume %c [range(%c, 0, {C - 1})] : index
{chr(10).join(reads)}
{chr(10).join(writes)}
    }}
  }}
  kernel.return
}}
""")
    # ---- q/k inverse L2 norms: inverse[t, 2*kh] (q), inverse[t, 2*kh + 1] (k)
    sq = "\n".join(f"""    %ix{m} = index.constant {32 * m} : index
    %d{m} = index.add %lane, %ix{m} : index
    %c{m}v = index.add %base, %d{m} : index
    %cc{m} = index.assume %c{m}v [range(%c{m}v, 0, {C - 1})] : index
    %e{m} = view.load %x_view[%token, %cc{m}] : view<[%tokens]x{C}xf32> -> f32
    %s{m} = scalar.fmaf %e{m}, %e{m}, {"%zero_f" if m == 0 else f"%s{m - 1}"} : f32""" for m in range(RK))
    T.append(f"""
kernel.def export("g_gdn_qk_norm") @g_gdn_qk_norm(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %x: buffer, %inverse: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, 7)] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c8 = index.constant 8 : index
  %chk = index.constant {HK} : index
  %heads = index.constant {2 * HK} : index
  %cdk = index.constant {DK} : index
  %zero_f = scalar.constant 0.0 : f32
  %eps = scalar.constant {eps!r} : f32
  %x_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x{C}xf32>
  %inv_view = buffer.view %inverse[%zero] : buffer -> view<[%tokens]x{2 * HK}xf32>
  scf.for %head = [%wave to %heads step %c8] {{
    %h = index.assume %head [range(%head, 0, {2 * HK - 1})] : index
    %base = index.mul %h, %cdk : index
{sq}
    %sum = kernel.subgroup.reduce<addf> %s{RK - 1} : f32
    %stable = scalar.addf %sum, %eps : f32
    %inv = scalar.rsqrtf %stable : f32
    %is_q = index.cmp ult, %h, %chk : index
    %kh_k = index.sub %h, %chk : index
    %kh = scf.select %is_q, %h, %kh_k : index
    %c2 = index.constant 2 : index
    %slot_a = index.mul %kh, %c2 : index
    %c1 = index.constant 1 : index
    %slot_k = index.add %slot_a, %c1 : index
    %slot_v = scf.select %is_q, %slot_a, %slot_k : index
    %slot = index.assume %slot_v [range(%slot_v, 0, {2 * HK - 1})] : index
    %leader = index.cmp eq, %lane, %c0 : index
    scf.if %leader {{
      view.store %inv, %inv_view[%token, %slot] : f32, view<[%tokens]x{2 * HK}xf32>
    }}
  }}
  kernel.return
}}
""")
    # ---- recurrence: workgroup (value head, 4-column tile, span) of 32 lanes; lane owns key rows lane + 32m
    loads = []
    for m in range(RK):
        loads.append(f"""    %qr{m}c = index.constant {32 * m} : index
    %qr{m} = index.add %q_row0, %qr{m}c : index
    %kr{m} = index.add %k_row0, %qr{m}c : index
    %q{m}_raw = view.load %x_view[%tabs, %qr{m}] : view<[%tokens]x{C}xf32> -> f32
    %k{m}_raw = view.load %x_view[%tabs, %kr{m}] : view<[%tokens]x{C}xf32> -> f32
    %q{m} = scalar.mulf %q{m}_raw, %q_inverse : f32
    %k{m} = scalar.mulf %k{m}_raw, %k_inverse : f32
    %qv{m} = vector.splat %q{m} : vector<4xf32>
    %kv{m} = vector.splat %k{m} : vector<4xf32>
    %dec{m} = vector.mulf %old{m}, %decay_vector : vector<4xf32>
    %pr{m} = vector.mulf %dec{m}, %kv{m} : vector<4xf32>""")
    psum = "%pr0"
    for m in range(1, RK):
        loads.append(f"    %ps{m} = vector.addf {psum}, %pr{m} : vector<4xf32>"); psum = f"%ps{m}"
    upd = []
    for m in range(RK):
        upd.append(f"""    %up{m} = vector.mulf %kv{m}, %delta : vector<4xf32>
    %new{m} = vector.addf %dec{m}, %up{m} : vector<4xf32>
    %ot{m} = vector.mulf %new{m}, %qv{m} : vector<4xf32>""")
    osum = "%ot0"
    for m in range(1, RK):
        upd.append(f"    %os{m} = vector.addf {osum}, %ot{m} : vector<4xf32>"); osum = f"%os{m}"
    state_idx = "\n".join(f"""  %si{m}c = index.constant {32 * m} : index
  %si{m} = index.add %state_index0, %si{m}c : index""" for m in range(1, RK))
    state_idx = "  %si0 = index.add %state_index0, %c0 : index\n" + state_idx
    gathers = "\n".join(f"""      %g{m} = vector.gather %state_view[%si{m}][%column_offsets] : view<{HV * DV * DK}xf32>, vector<4xindex> -> vector<4xf32>""" for m in range(RK))
    scatters = "\n".join(f"""    vector.scatter %final{m}, %next_view[%si{m}][%column_offsets] : vector<4xf32>, view<{HV * DV * DK}xf32>, vector<4xindex>""" for m in range(RK))
    v4 = ", ".join(["vector<4xf32>"] * RK)
    T.append(f"""
kernel.def export("g_gdn_recurrence") @g_gdn_recurrence(%spans: index) {{
  %c1 = index.constant 1 : index
  %heads = index.constant {HV} : index
  %tiles = index.constant {DV // 4} : index
  %c32 = index.constant 32 : index
  kernel.launch.config workgroups(%heads, %tiles, %spans) workgroup_size(%c32, %c1, %c1) : index
}} launch(%tokens_arg: index, %spans_arg: index, %layer_arg: index, %metadata: buffer, %origins: buffer, %arena: buffer, %x: buffer, %inverse: buffer, %alpha: buffer, %beta: buffer, %ssm_a: buffer, %dt_bias: buffer, %out: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %layer = index.assume %layer_arg [range(%layer_arg, 0, {NG - 1})] : index
  %head_id = kernel.workgroup.id<x> : index
  %vh = index.assume %head_id [range(%head_id, 0, {HV - 1})] : index
  %tile_id = kernel.workgroup.id<y> : index
  %tile = index.assume %tile_id [range(%tile_id, 0, {DV // 4 - 1})] : index
  %span_id = kernel.workgroup.id<z> : index
  %span = index.assume %span_id [lt(%span_id, %spans)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1i = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %zero_f = scalar.constant 0.0 : f32
  %active, %length, %start32, %row, %first = func.call @g_gdn_span(%metadata, %spans, %span) : (buffer, index, index) -> (i1, index, i32, index, index)
  scf.if %active {{
  %layer_stride = index.constant {LB} : offset
  %layer_bytes = index.scale %layer, %layer_stride : index, offset -> offset
  %reader, %writer, %fresh = func.call @g_gdn_slots(%origins, %row, %start32, %layer_bytes) : (buffer, index, i32, offset) -> (offset, offset, i1)
  %x_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x{C}xf32>
  %inv_view = buffer.view %inverse[%zero] : buffer -> view<[%tokens]x{2 * HK}xf32>
  %alpha_view = buffer.view %alpha[%zero] : buffer -> view<[%tokens]x{HV}xf32>
  %beta_view = buffer.view %beta[%zero] : buffer -> view<[%tokens]x{HV}xf32>
  %a_view = buffer.view %ssm_a[%zero] : buffer -> view<{HV}xf32>
  %bias_view = buffer.view %dt_bias[%zero] : buffer -> view<{HV}xf32>
  %out_view = buffer.view %out[%zero] : buffer -> view<[%tokens]x{VD}xf32>
  %state_view = buffer.view %arena[%reader] : buffer -> view<{HV * DV * DK}xf32>
  %next_view = buffer.view %arena[%writer] : buffer -> view<{HV * DV * DK}xf32>
  %cdk = index.constant {DK} : index
  %cdv = index.constant {DV} : index
  %chk = index.constant {HK} : index
  %column_offsets = vector.iota %c0 step %cdk : vector<4xindex>
  %kh = index.rem %vh, %chk : index
  %kh_base = index.mul %kh, %cdk : index
  %q_row0 = index.add %kh_base, %lane : index
  %k_skip = index.constant {HK * DK} : index
  %k_row0 = index.add %q_row0, %k_skip : index
  %q_norm_index = index.mul %kh, %c2 : index
  %k_norm_index = index.add %q_norm_index, %c1i : index
  %column_base = index.mul %tile, %c4 : index
  %head_columns = index.mul %vh, %cdv : index
  %state_column = index.add %head_columns, %column_base : index
  %state_column_base = index.mul %state_column, %cdk : index
  %state_index0 = index.add %state_column_base, %lane : index
{state_idx}
  %zero4 = vector.splat %zero_f : vector<4xf32>
  {", ".join(f"%init{m}" for m in range(RK))} = scf.if %fresh -> ({v4}) {{
    scf.yield {", ".join(["%zero4"] * RK)} : {v4}
  }} else {{
{gathers}
    scf.yield {", ".join(f"%g{m}" for m in range(RK))} : {v4}
  }}
  %a = view.load %a_view[%vh] : view<{HV}xf32> -> f32
  %dt_bias_value = view.load %bias_view[%vh] : view<{HV}xf32> -> f32
  %v_skip = index.constant {2 * HK * DK} : index
  %v_head = index.add %v_skip, %head_columns : index
  %v_channel = index.add %v_head, %column_base : index
  %out_channel = index.add %head_columns, %column_base : index
  %scale = scalar.constant {DK ** -0.5!r} : f32
  %scale_v = vector.splat %scale : vector<4xf32>
  {", ".join(f"%final{m}" for m in range(RK))} = scf.for %step = [%c0 to %length step %c1i]({", ".join(f"%old{m} = %init{m} : vector<4xf32>" for m in range(RK))}) -> ({v4}) {{
    %tabs_value = index.add %first, %step : index
    %tabs = index.assume %tabs_value [lt(%tabs_value, %tokens)] : index
    %q_inverse = view.load %inv_view[%tabs, %q_norm_index] : view<[%tokens]x{2 * HK}xf32> -> f32
    %k_inverse = view.load %inv_view[%tabs, %k_norm_index] : view<[%tokens]x{2 * HK}xf32> -> f32
    %alpha_raw = view.load %alpha_view[%tabs, %vh] : view<[%tokens]x{HV}xf32> -> f32
    %alpha_biased = scalar.addf %alpha_raw, %dt_bias_value : f32
    %alpha_positive = scalar.softplusf<afn> %alpha_biased : f32
    %gate = scalar.mulf %alpha_positive, %a : f32
    %decay = scalar.expf<afn> %gate : f32
    %decay_vector = vector.splat %decay : vector<4xf32>
{chr(10).join(loads)}
    %prediction = kernel.workgroup.reduce<addf> {psum} : vector<4xf32>
    %value = vector.load %x_view[%tabs, %v_channel] : view<[%tokens]x{C}xf32> -> vector<4xf32>
    %residual = vector.subf %value, %prediction : vector<4xf32>
    %beta_raw = view.load %beta_view[%tabs, %vh] : view<[%tokens]x{HV}xf32> -> f32
    %beta_value = scalar.logisticf<afn> %beta_raw : f32
    %beta_vector = vector.splat %beta_value : vector<4xf32>
    %delta = vector.mulf %residual, %beta_vector : vector<4xf32>
{chr(10).join(upd)}
    %output = kernel.workgroup.reduce<addf> {osum} : vector<4xf32>
    %leader = index.cmp eq, %lane, %c0 : index
    scf.if %leader {{
      %scaled = vector.mulf %output, %scale_v : vector<4xf32>
      vector.store %scaled, %out_view[%tabs, %out_channel] : vector<4xf32>, view<[%tokens]x{VD}xf32>
    }}
    scf.yield {", ".join(f"%new{m}" for m in range(RK))} : {v4}
  }}
{scatters}
  }}
  kernel.return
}}
""")
    # ---- gated RMSNorm per value head: out = rmsnorm(o) * w * silu(z); padding tokens get zeros
    nl = "\n".join(f"""    %ni{m}c = index.constant {32 * m} : index
    %nd{m} = index.add %lane, %ni{m}c : index
    %nc{m}v = index.add %hbase, %nd{m} : index
    %nc{m} = index.assume %nc{m}v [range(%nc{m}v, 0, {VD - 1})] : index
    %r{m} = view.load %raw_view[%token, %nc{m}] : view<[%tokens]x{VD}xf32> -> f32
    %ss{m} = scalar.fmaf %r{m}, %r{m}, {"%zero_f" if m == 0 else f"%ss{m - 1}"} : f32""" for m in range(RV))
    nw = "\n".join(f"""    %wd{m} = view.load %w_view[%nd{m}] : view<{DV}xf32> -> f32
    %z{m} = view.load %z_view[%token, %nc{m}] : view<[%tokens]x{VD}xf32> -> f32
    %zs{m} = scalar.logisticf<afn> %z{m} : f32
    %zg{m} = scalar.mulf %z{m}, %zs{m} : f32
    %n{m}a = scalar.mulf %r{m}, %inv : f32
    %n{m}b = scalar.mulf %n{m}a, %wd{m} : f32
    %n{m} = scalar.mulf %n{m}b, %zg{m} : f32
    %res{m} = scf.select %live, %n{m}, %zero_f : f32
    view.store %res{m}, %o_view[%token, %nc{m}] : f32, view<[%tokens]x{VD}xf32>""" for m in range(RV))
    T.append(f"""
kernel.def export("g_gdn_norm_gate") @g_gdn_norm_gate(%tokens: index) {{
  %c1 = index.constant 1 : index
  %heads = index.constant {HV} : index
  %c32 = index.constant 32 : index
  kernel.launch.config workgroups(%tokens, %heads, %c1) workgroup_size(%c32, %c1, %c1) : index
}} launch(%tokens_arg: index, %info: buffer, %raw: buffer, %weight: buffer, %z: buffer, %o: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %head_id = kernel.workgroup.id<y> : index
  %vh = index.assume %head_id [range(%head_id, 0, {HV - 1})] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0_i32 = scalar.constant 0 : i32
  %zero_f = scalar.constant 0.0 : f32
  %eps = scalar.constant {eps!r} : f32
  %count = scalar.constant {float(DV)!r} : f32
  %cdv = index.constant {DV} : index
  %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %live32 = view.load %info_view[%token, 2] : view<[%tokens]x4xi32> -> i32
  %live = scalar.cmpi ne, %live32, %c0_i32 : i32
  %raw_view = buffer.view %raw[%zero] : buffer -> view<[%tokens]x{VD}xf32>
  %z_view = buffer.view %z[%zero] : buffer -> view<[%tokens]x{VD}xf32>
  %o_view = buffer.view %o[%zero] : buffer -> view<[%tokens]x{VD}xf32>
  %w_view = buffer.view %weight[%zero] : buffer -> view<{DV}xf32>
  %hbase = index.mul %vh, %cdv : index
{nl}
    %sum = kernel.subgroup.reduce<addf> %ss{RV - 1} : f32
    %mean = scalar.divf %sum, %count : f32
    %stable = scalar.addf %mean, %eps : f32
    %inv = scalar.rsqrtf %stable : f32
{nw}
  kernel.return
}}
""")
    return "\n".join(T)


SPLIT_QG = r"""
// qwen35-style attention: attn_q packs [query (D) | gate (D)] per head; split into q and gate.
kernel.def export("g_split_qgate") @g_split_qgate(%tokens: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%tokens_arg: index, %qg: buffer, %q: buffer, %gate: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %thread_id = kernel.workitem.id<x> : index
  %thread = index.assume %thread_id [range(%thread_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %cqd = index.constant @@QD@@ : index
  %cd = index.constant @@D@@ : index
  %c256 = index.constant 256 : index
  %qg_view = buffer.view %qg[%zero] : buffer -> view<[%tokens]x@@QH@@x2x@@D@@xf32>
  %q_view = buffer.view %q[%zero] : buffer -> view<[%tokens]x@@QH@@x@@D@@xf32>
  %g_view = buffer.view %gate[%zero] : buffer -> view<[%tokens]x@@QH@@x@@D@@xf32>
  scf.for %i = [%thread to %cqd step %c256] {
    %bi = index.assume %i [range(%i, 0, @@QD_LESS@@)] : index
    %h_value = index.div %bi, %cd : index
    %h = index.assume %h_value [range(%h_value, 0, @@QH_LESS@@)] : index
    %d_value = index.rem %bi, %cd : index
    %d = index.assume %d_value [range(%d_value, 0, @@D_LESS@@)] : index
    %qv = view.load %qg_view[%token, %h, 0, %d] : view<[%tokens]x@@QH@@x2x@@D@@xf32> -> f32
    %gv = view.load %qg_view[%token, %h, 1, %d] : view<[%tokens]x@@QH@@x2x@@D@@xf32> -> f32
    view.store %qv, %q_view[%token, %h, %d] : f32, view<[%tokens]x@@QH@@x@@D@@xf32>
    view.store %gv, %g_view[%token, %h, %d] : f32, view<[%tokens]x@@QH@@x@@D@@xf32>
  }
  kernel.return
}
"""


def split_qgate(QH, D):
    return (SPLIT_QG.replace("@@QD_LESS@@", str(QH * D - 1)).replace("@@QH_LESS@@", str(QH - 1)).replace("@@D_LESS@@", str(D - 1))
            .replace("@@QD@@", str(QH * D)).replace("@@QH@@", str(QH)).replace("@@D@@", str(D)))
