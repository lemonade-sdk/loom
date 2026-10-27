# Generates llm.loom: kernel library + forward-pass command program for a dense
# decoder-only LLM stored as GGUF, driven by a config JSON from inspect_gguf.py.
#
# Generalized from the Muse Glimmer 30B port; end-to-end validated (llama.cpp
# golden, 1k-32k prompts) on Qwen3-0.6B. Supported:
#   * head_dim 128, any query/KV head counts with heads % kv_heads == 0
#   * GGUF weight formats q4_k, q5_k, q6_k, q8_0 (mixed per tensor is fine)
#   * RMSNorm, SwiGLU FFN, NeoX or interleaved ("norm") RoPE, optional per-head
#     Q/K RMSNorm, optional per-layer sliding window, optional sigmoid attention
#     output gate and post-attention/post-FFN norms (Glimmer), optional
#     embedding RMSNorm, tied or separate output head.
# Greedy decoding only: the head computes an argmax (logit scale/softcap are
# monotonic and therefore skipped).
import json, math, os, subprocess, sys

CFG = json.load(open(os.environ.get("LLM_CONFIG") or sys.argv[1]))
H = CFG["hidden"]; HG = H // 32          # hidden size, 32-value groups per hidden row
QH, KVH, D = CFG["heads"], CFG["kv_heads"], CFG["head_dim"]
FF, V, L = CFG["ffn"], CFG["vocab"], CFG["layers"]
G = QH // KVH                             # query heads per KV head (waves per attention workgroup)
assert D == 128, "kernels assume head_dim 128 (lane l owns dims 4l..4l+3 / pairs of a 64-lane half)"
assert QH % KVH == 0 and G <= 16, "need heads % kv_heads == 0 and at most 16 query heads per KV head"
assert HG <= 256 and H % 32 == 0, "hidden must be a multiple of 32 and at most 8192"
UNROLL = 16
POS_BYTES = KVH * D * 2                   # F16 K (or V) bytes per position
WINDOW = CFG.get("sliding_window") or 0   # keys kept by sliding layers (0: none)
GATE = CFG.get("attn_gate", False)
POST_NORM = CFG.get("post_norms", False)
POST_EPS = CFG.get("post_norm_eps", 1e-8)
EMBED_NORM = CFG.get("embed_norm", False)
EPS = CFG["rms_eps"]
QK_NORM = CFG.get("qk_norm", False)
ROPE = CFG.get("rope_type", "neox")       # "neox": pairs (d, d+64); "norm": pairs (2d, 2d+1)
THETA = CFG["rope_theta"]
TENS = CFG["tensors"]                     # tensor name -> {"type": fmt, "shape": [...]}

FMT = {  # bytes per superblock of SB[fmt] values
    "q4_k": 144, "q5_k": 176, "q6_k": 210, "q8_0": 34,
}
SB = {"q4_k": 256, "q5_k": 256, "q6_k": 256, "q8_0": 32}


def row_bytes(fmt, k):
    return k // SB[fmt] * FMT[fmt]


def fmt_of(name):
    return TENS[name]["type"]


def tensor_bytes(name):
    t = TENS[name]
    n = 1
    for d in t["shape"]:
        n *= d
    if t["type"] == "f32":
        return n * 4
    return n // SB[t["type"]] * FMT[t["type"]]


COMMON = r"""
// Paged KV: 2 planes per layer (layer-major, K then V); one position holds the
// KV heads x 128 F16 values in 16-position pages mapped per session row.
func.def inline @g_kv_offset(%origins: buffer, %row: index, %position: index, %blocks: index, %plane: index, %pool: index) -> (offset) {
  %c1 = index.constant 1 : index
  %c16 = index.constant 16 : index
  %record_stride = index.constant 40 : offset
  %position_bytes = index.constant 512 : offset
  %bounded_row = index.assume %row [range(%row, 0, 2147483646)] : index
  %bounded_position = index.assume %position [range(%position, 0, 262143)] : index
  %record_offset = index.scale %bounded_row, %record_stride : index, offset -> offset
  %records = buffer.view %origins[%record_offset] : buffer -> view<5xi64>
  %map64 = view.load %records[4] : view<5xi64> -> i64
  %map_bound = scalar.assume %map64 [range(%map64, 0, 9223372036854775807)] : i64
  %map_origin = index.cast %map_bound : i64 to offset
  %page_value = index.div %bounded_position, %c16 : index
  %last = index.sub %blocks, %c1 : index
  %page = index.assume %page_value [range(%page_value, 0, %last), lt(%page_value, %blocks)] : index
  %map = buffer.view %origins[%map_origin] : buffer -> view<[%blocks]xi32>
  %physical32 = view.load %map[%page] : view<[%blocks]xi32> -> i32
  %bounded_physical = scalar.assume %physical32 [range(%physical32, 0, 1048575)] : i32
  %physical = index.cast %bounded_physical : i32 to index
  %page_base = index.mul %physical, %c16 : index
  %within = index.rem %bounded_position, %c16 : index
  %slot = index.add %page_base, %within : index
  %plane_slot = index.madd %plane, %pool, %slot : index
  %bytes = index.scale %plane_slot, %position_bytes : index, offset -> offset
  func.return %bytes : offset
}

// Q4_K / Q5_K 6-bit scale and minimum of group `group` (0-7) of a superblock,
// from hrx-loom-library motif/format/ggml/q4_k.loom.
func.def inline @g_k_scale_minimum(%packed_scales: vector<3xi32>, %group: index) -> (i32, i32) {
  %bounded_group = index.assume %group [range(%group, 0, 7)] : index
  %scale0 = vector.extract %packed_scales[0] : vector<3xi32> -> i32
  %scale1 = vector.extract %packed_scales[1] : vector<3xi32> -> i32
  %scale2 = vector.extract %packed_scales[2] : vector<3xi32> -> i32
  %c4 = index.constant 4 : index
  %c8 = index.constant 8 : index
  %c2_i32 = scalar.constant 2 : i32
  %c4_i32 = scalar.constant 4 : i32
  %c15_i32 = scalar.constant 15 : i32
  %c48_i32 = scalar.constant 48 : i32
  %is_low_group = index.cmp ult, %bounded_group, %c4 : index
  %lane = index.rem %bounded_group, %c4 : index
  %shift_index = index.mul %lane, %c8 : index
  %shift = index.cast %shift_index : index to i32
  %high_shift = scalar.addi %shift, %c2_i32 : i32
  %minimum_shift = scalar.addi %shift, %c4_i32 : i32
  %scale_low_source = scf.select %is_low_group, %scale0, %scale2 : i32
  %minimum_low_source = scf.select %is_low_group, %scale1, %scale2 : i32
  %scale_high_shift = scf.select %is_low_group, %shift, %high_shift : i32
  %minimum_low_shift = scf.select %is_low_group, %shift, %minimum_shift : i32
  %scale_low_shifted = scalar.shrui %scale_low_source, %shift : i32
  %scale_low = scalar.andi %scale_low_shifted, %c15_i32 : i32
  %scale_high_shifted = scalar.shrui %scale0, %scale_high_shift : i32
  %scale_high = scalar.andi %scale_high_shifted, %c48_i32 : i32
  %scale = scalar.ori %scale_low, %scale_high : i32
  %minimum_low_shifted = scalar.shrui %minimum_low_source, %minimum_low_shift : i32
  %minimum_low = scalar.andi %minimum_low_shifted, %c15_i32 : i32
  %minimum_high_shifted = scalar.shrui %scale1, %scale_high_shift : i32
  %minimum_high = scalar.andi %minimum_high_shifted, %c48_i32 : i32
  %minimum = scalar.ori %minimum_low, %minimum_high : i32
  func.return %scale, %minimum : i32, i32
}

// (d * scale, dmin * minimum) of group `group` from a 16-byte K-quant header.
func.def inline @g_k_metadata(%header: vector<4xi32>, %group: index) -> (f32, f32) {
  %halves = vector.bitcast %header : vector<4xi32> to vector<8xf16>
  %d_half = vector.extract %halves[0] : vector<8xf16> -> f16
  %minimum_half = vector.extract %halves[1] : vector<8xf16> -> f16
  %d = scalar.extf %d_half : f16 to f32
  %dmin = scalar.extf %minimum_half : f16 to f32
  %first = vector.extract %header[1] : vector<4xi32> -> i32
  %second = vector.extract %header[2] : vector<4xi32> -> i32
  %third = vector.extract %header[3] : vector<4xi32> -> i32
  %packed = vector.from_elements %first, %second, %third : vector<3xi32>
  %scale, %minimum = func.call @g_k_scale_minimum(%packed, %group) : (vector<3xi32>, index) -> (i32, i32)
  %scale_float = scalar.uitofp %scale : i32 to f32
  %minimum_float = scalar.uitofp %minimum : i32 to f32
  %scaled = scalar.mulf %d, %scale_float : f32
  %minimum_scaled = scalar.mulf %dmin, %minimum_float : f32
  func.return %scaled, %minimum_scaled : f32, f32
}

// The 32 values of group `group` (0-7) of a Q4_K superblock at `origin`.
func.def inline @g_q4_k_values(%w: buffer, %origin: offset, %group: index) -> (vector<32xf32>) {
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %header_bytes = index.constant 16 : offset
  %page_bytes = index.constant 32 : offset
  %nibble_mask = vector.constant 252645135 : vector<8xi32>
  %header_view = buffer.view %w[%origin] : buffer -> view<4xi32>
  %header = vector.load %header_view[0] : view<4xi32> -> vector<4xi32>
  %scale, %minimum = func.call @g_k_metadata(%header, %group) : (vector<4xi32>, index) -> (f32, f32)
  %page = index.div %group, %c2 : index
  %nibble = index.rem %group, %c2 : index
  %page_add = index.scale %page, %page_bytes : index, offset -> offset
  %payload = index.add %origin, %header_bytes : offset
  %codes_origin = index.add %payload, %page_add : offset
  %code_view = buffer.view %w[%codes_origin] : buffer -> view<8xi32>
  %packed = vector.load %code_view[0] : view<8xi32> -> vector<8xi32>
  %shift_bits = index.mul %nibble, %c4 : index
  %shift_i32 = index.cast %shift_bits : index to i32
  %shift = vector.splat %shift_i32 : vector<8xi32>
  %shifted = vector.shrui %packed, %shift : vector<8xi32>
  %masked = vector.andi %shifted, %nibble_mask : vector<8xi32>
  %bytes = vector.bitcast %masked : vector<8xi32> to vector<32xi8>
  %codes = vector.uitofp %bytes : vector<32xi8> to vector<32xf32>
  %scales = vector.splat %scale : vector<32xf32>
  %minima = vector.splat %minimum : vector<32xf32>
  %scaled = vector.mulf %codes, %scales : vector<32xf32>
  %values = vector.subf %scaled, %minima : vector<32xf32>
  func.return %values : vector<32xf32>
}

// Dot of group `group` of a Q4_K superblock with 32 inputs.
func.def inline @g_q4_k_dot(%w: buffer, %origin: offset, %group: index, %inputs: vector<32xf32>) -> (f32) {
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %header_bytes = index.constant 16 : offset
  %page_bytes = index.constant 32 : offset
  %zero_f = scalar.constant 0.0 : f32
  %nibble_mask = vector.constant 252645135 : vector<8xi32>
  %header_view = buffer.view %w[%origin] : buffer -> view<4xi32>
  %header = vector.load %header_view[0] : view<4xi32> -> vector<4xi32>
  %scale, %minimum = func.call @g_k_metadata(%header, %group) : (vector<4xi32>, index) -> (f32, f32)
  %page = index.div %group, %c2 : index
  %nibble = index.rem %group, %c2 : index
  %page_add = index.scale %page, %page_bytes : index, offset -> offset
  %payload = index.add %origin, %header_bytes : offset
  %codes_origin = index.add %payload, %page_add : offset
  %code_view = buffer.view %w[%codes_origin] : buffer -> view<8xi32>
  %packed = vector.load %code_view[0] : view<8xi32> -> vector<8xi32>
  %shift_bits = index.mul %nibble, %c4 : index
  %shift_i32 = index.cast %shift_bits : index to i32
  %shift = vector.splat %shift_i32 : vector<8xi32>
  %shifted = vector.shrui %packed, %shift : vector<8xi32>
  %masked = vector.andi %shifted, %nibble_mask : vector<8xi32>
  %bytes = vector.bitcast %masked : vector<8xi32> to vector<32xi8>
  %codes = vector.uitofp %bytes : vector<32xi8> to vector<32xf32>
  %products = vector.mulf %codes, %inputs : vector<32xf32>
  %qx = vector.reduce<addf> %products, %zero_f : vector<32xf32>, f32
  %xs = vector.reduce<addf> %inputs, %zero_f : vector<32xf32>, f32
  %scaled = scalar.mulf %qx, %scale : f32
  %offset_term = scalar.mulf %xs, %minimum : f32
  %dot = scalar.subf %scaled, %offset_term : f32
  func.return %dot : f32
}

// Q5_K: header (d, dmin, 12 scale bytes), 32 high-bit bytes, 128 nibble bytes.
// Value of group g: d*scale_g * (nibble | (qh_bit_g << 4)) - dmin*min_g.
func.def inline @g_q5_k_dot(%w: buffer, %origin: offset, %group: index, %inputs: vector<32xf32>) -> (f32) {
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %high_bytes = index.constant 16 : offset
  %low_bytes = index.constant 48 : offset
  %page_bytes = index.constant 32 : offset
  %zero_f = scalar.constant 0.0 : f32
  %nibble_mask = vector.constant 252645135 : vector<8xi32>
  %bit_mask = vector.constant 16843009 : vector<8xi32>
  %four = vector.constant 4 : vector<8xi32>
  %header_view = buffer.view %w[%origin] : buffer -> view<4xi32>
  %header = vector.load %header_view[0] : view<4xi32> -> vector<4xi32>
  %scale, %minimum = func.call @g_k_metadata(%header, %group) : (vector<4xi32>, index) -> (f32, f32)
  %page = index.div %group, %c2 : index
  %nibble = index.rem %group, %c2 : index
  %page_add = index.scale %page, %page_bytes : index, offset -> offset
  %payload = index.add %origin, %low_bytes : offset
  %codes_origin = index.add %payload, %page_add : offset
  %code_view = buffer.view %w[%codes_origin] : buffer -> view<8xi32>
  %packed = vector.load %code_view[0] : view<8xi32> -> vector<8xi32>
  %high_origin = index.add %origin, %high_bytes : offset
  %high_view = buffer.view %w[%high_origin] : buffer -> view<8xi32>
  %high_packed = vector.load %high_view[0] : view<8xi32> -> vector<8xi32>
  %shift_bits = index.mul %nibble, %c4 : index
  %shift_i32 = index.cast %shift_bits : index to i32
  %shift = vector.splat %shift_i32 : vector<8xi32>
  %shifted = vector.shrui %packed, %shift : vector<8xi32>
  %low = vector.andi %shifted, %nibble_mask : vector<8xi32>
  %group_i32 = index.cast %group : index to i32
  %group_shift = vector.splat %group_i32 : vector<8xi32>
  %high_shifted = vector.shrui %high_packed, %group_shift : vector<8xi32>
  %high_bit = vector.andi %high_shifted, %bit_mask : vector<8xi32>
  %high = vector.shli %high_bit, %four : vector<8xi32>
  %combined = vector.ori %low, %high : vector<8xi32>
  %bytes = vector.bitcast %combined : vector<8xi32> to vector<32xi8>
  %codes = vector.uitofp %bytes : vector<32xi8> to vector<32xf32>
  %products = vector.mulf %codes, %inputs : vector<32xf32>
  %qx = vector.reduce<addf> %products, %zero_f : vector<32xf32>, f32
  %xs = vector.reduce<addf> %inputs, %zero_f : vector<32xf32>, f32
  %scaled = scalar.mulf %qx, %scale : f32
  %offset_term = scalar.mulf %xs, %minimum : f32
  %dot = scalar.subf %scaled, %offset_term : f32
  func.return %dot : f32
}

// Q6_K (210-byte superblock): ql[128] low nibbles, qh[64] high 2 bits,
// 16 int8 scales, F16 d. Group g = 4*half + j covers values half*128 + j*32
// + l: q = ((ql[half*64 + (j%2)*32 + l] >> 4*(j/2)) & 15)
//        | (((qh[half*32 + l] >> 2*j) & 3) << 4), value = d * sc[half*8 + 2j
// + l/16] * (q - 32). Superblocks are only 2-byte aligned, so codes load as bytes.
func.def inline @g_q6_k_dot(%w: buffer, %origin: offset, %group: index, %inputs: vector<32xf32>) -> (f32) {
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c32 = index.constant 32 : index
  %c64 = index.constant 64 : index
  %qh_base = index.constant 128 : index
  %sc_base = index.constant 192 : index
  %d_bytes = index.constant 208 : offset
  %zero_f = scalar.constant 0.0 : f32
  %thirty_two = scalar.constant 32.0 : f32
  %low_mask = vector.constant 15 : vector<32xi8>
  %two_bits = vector.constant 3 : vector<32xi8>
  %four8 = vector.constant 4 : vector<32xi8>
  %bounded = index.assume %group [range(%group, 0, 7)] : index
  %half = index.div %bounded, %c4 : index
  %j = index.rem %bounded, %c4 : index
  %j_low = index.rem %j, %c2 : index
  %j_high = index.div %j, %c2 : index
  %ql_half = index.mul %half, %c64 : index
  %ql_part = index.mul %j_low, %c32 : index
  %ql_index = index.add %ql_half, %ql_part : index
  %qh_half = index.mul %half, %c32 : index
  %qh_index = index.add %qh_base, %qh_half : index
  %sc_half = index.mul %half, %c8_q6 : index
  %sc_j = index.mul %j, %c2 : index
  %sc_part = index.add %sc_half, %sc_j : index
  %sc_index = index.add %sc_base, %sc_part : index
  %ql_add = index.cast %ql_index : index to offset
  %qh_add = index.cast %qh_index : index to offset
  %sc_add = index.cast %sc_index : index to offset
  %ql_origin = index.add %origin, %ql_add : offset
  %qh_origin = index.add %origin, %qh_add : offset
  %sc_origin = index.add %origin, %sc_add : offset
  %d_origin = index.add %origin, %d_bytes : offset
  %ql_view = buffer.view %w[%ql_origin] : buffer -> view<32xi8>
  %qh_view = buffer.view %w[%qh_origin] : buffer -> view<32xi8>
  %sc_view = buffer.view %w[%sc_origin] : buffer -> view<2xi8>
  %d_view = buffer.view %w[%d_origin] : buffer -> view<1xf16>
  %ql = vector.load %ql_view[0] : view<32xi8> -> vector<32xi8>
  %qh = vector.load %qh_view[0] : view<32xi8> -> vector<32xi8>
  %sc = vector.load %sc_view[0] : view<2xi8> -> vector<2xi8>
  %d_half = view.load %d_view[0] : view<1xf16> -> f16
  %d = scalar.extf %d_half : f16 to f32
  %shift_low = index.mul %j_high, %c4 : index
  %shift_high = index.mul %j, %c2 : index
  %shift_low32 = index.cast %shift_low : index to i32
  %shift_high32 = index.cast %shift_high : index to i32
  %shift_low8 = scalar.trunci %shift_low32 : i32 to i8
  %shift_high8 = scalar.trunci %shift_high32 : i32 to i8
  %shift_low_v = vector.splat %shift_low8 : vector<32xi8>
  %shift_high_v = vector.splat %shift_high8 : vector<32xi8>
  %ql_shifted = vector.shrui %ql, %shift_low_v : vector<32xi8>
  %low = vector.andi %ql_shifted, %low_mask : vector<32xi8>
  %qh_shifted = vector.shrui %qh, %shift_high_v : vector<32xi8>
  %qh_bits = vector.andi %qh_shifted, %two_bits : vector<32xi8>
  %high = vector.shli %qh_bits, %four8 : vector<32xi8>
  %combined = vector.ori %low, %high : vector<32xi8>
  %codes = vector.uitofp %combined : vector<32xi8> to vector<32xf32>
  %s0_i8 = vector.extract %sc[0] : vector<2xi8> -> i8
  %s1_i8 = vector.extract %sc[1] : vector<2xi8> -> i8
  %s0 = scalar.sitofp %s0_i8 : i8 to f32
  %s1 = scalar.sitofp %s1_i8 : i8 to f32
  %s0_v = vector.splat %s0 : vector<16xf32>
  %s1_v = vector.splat %s1 : vector<16xf32>
  %scale_v = vector.concat<0> %s0_v, %s1_v : vector<16xf32>, vector<16xf32> -> vector<32xf32>
  %weighted = vector.mulf %inputs, %scale_v : vector<32xf32>
  %products = vector.mulf %codes, %weighted : vector<32xf32>
  %qx = vector.reduce<addf> %products, %zero_f : vector<32xf32>, f32
  %xs = vector.reduce<addf> %weighted, %zero_f : vector<32xf32>, f32
  %offset_term = scalar.mulf %xs, %thirty_two : f32
  %centered = scalar.subf %qx, %offset_term : f32
  %dot = scalar.mulf %centered, %d : f32
  func.return %dot : f32
}

// Group `group` of a Q6_K superblock as 32 F32 values.
func.def inline @g_q6_k_values(%w: buffer, %origin: offset, %group: index) -> (vector<32xf32>) {
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %c8 = index.constant 8 : index
  %c32 = index.constant 32 : index
  %c64 = index.constant 64 : index
  %qh_base = index.constant 128 : index
  %sc_base = index.constant 192 : index
  %d_bytes = index.constant 208 : offset
  %thirty_two = vector.constant 32.0 : vector<32xf32>
  %low_mask = vector.constant 15 : vector<32xi8>
  %two_bits = vector.constant 3 : vector<32xi8>
  %four8 = vector.constant 4 : vector<32xi8>
  %bounded = index.assume %group [range(%group, 0, 7)] : index
  %half = index.div %bounded, %c4 : index
  %j = index.rem %bounded, %c4 : index
  %j_low = index.rem %j, %c2 : index
  %j_high = index.div %j, %c2 : index
  %ql_half = index.mul %half, %c64 : index
  %ql_part = index.mul %j_low, %c32 : index
  %ql_index = index.add %ql_half, %ql_part : index
  %qh_half = index.mul %half, %c32 : index
  %qh_index = index.add %qh_base, %qh_half : index
  %sc_half = index.mul %half, %c8 : index
  %sc_j = index.mul %j, %c2 : index
  %sc_part = index.add %sc_half, %sc_j : index
  %sc_index = index.add %sc_base, %sc_part : index
  %ql_add = index.cast %ql_index : index to offset
  %qh_add = index.cast %qh_index : index to offset
  %sc_add = index.cast %sc_index : index to offset
  %ql_origin = index.add %origin, %ql_add : offset
  %qh_origin = index.add %origin, %qh_add : offset
  %sc_origin = index.add %origin, %sc_add : offset
  %d_origin = index.add %origin, %d_bytes : offset
  %ql_view = buffer.view %w[%ql_origin] : buffer -> view<32xi8>
  %qh_view = buffer.view %w[%qh_origin] : buffer -> view<32xi8>
  %sc_view = buffer.view %w[%sc_origin] : buffer -> view<2xi8>
  %d_view = buffer.view %w[%d_origin] : buffer -> view<1xf16>
  %ql = vector.load %ql_view[0] : view<32xi8> -> vector<32xi8>
  %qh = vector.load %qh_view[0] : view<32xi8> -> vector<32xi8>
  %sc = vector.load %sc_view[0] : view<2xi8> -> vector<2xi8>
  %d_half = view.load %d_view[0] : view<1xf16> -> f16
  %d = scalar.extf %d_half : f16 to f32
  %shift_low = index.mul %j_high, %c4 : index
  %shift_high = index.mul %j, %c2 : index
  %shift_low32 = index.cast %shift_low : index to i32
  %shift_high32 = index.cast %shift_high : index to i32
  %shift_low8 = scalar.trunci %shift_low32 : i32 to i8
  %shift_high8 = scalar.trunci %shift_high32 : i32 to i8
  %shift_low_v = vector.splat %shift_low8 : vector<32xi8>
  %shift_high_v = vector.splat %shift_high8 : vector<32xi8>
  %ql_shifted = vector.shrui %ql, %shift_low_v : vector<32xi8>
  %low = vector.andi %ql_shifted, %low_mask : vector<32xi8>
  %qh_shifted = vector.shrui %qh, %shift_high_v : vector<32xi8>
  %qh_bits = vector.andi %qh_shifted, %two_bits : vector<32xi8>
  %high = vector.shli %qh_bits, %four8 : vector<32xi8>
  %combined = vector.ori %low, %high : vector<32xi8>
  %codes = vector.uitofp %combined : vector<32xi8> to vector<32xf32>
  %centered = vector.subf %codes, %thirty_two : vector<32xf32>
  %s0_i8 = vector.extract %sc[0] : vector<2xi8> -> i8
  %s1_i8 = vector.extract %sc[1] : vector<2xi8> -> i8
  %s0f = scalar.sitofp %s0_i8 : i8 to f32
  %s1f = scalar.sitofp %s1_i8 : i8 to f32
  %s0 = scalar.mulf %s0f, %d : f32
  %s1 = scalar.mulf %s1f, %d : f32
  %s0_v = vector.splat %s0 : vector<16xf32>
  %s1_v = vector.splat %s1 : vector<16xf32>
  %scale_v = vector.concat<0> %s0_v, %s1_v : vector<16xf32>, vector<16xf32> -> vector<32xf32>
  %values = vector.mulf %centered, %scale_v : vector<32xf32>
  func.return %values : vector<32xf32>
}

// Group `group` of a Q5_K superblock as 32 F32 values.
func.def inline @g_q5_k_values(%w: buffer, %origin: offset, %group: index) -> (vector<32xf32>) {
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %high_bytes = index.constant 16 : offset
  %low_bytes = index.constant 48 : offset
  %page_bytes = index.constant 32 : offset
  %nibble_mask = vector.constant 252645135 : vector<8xi32>
  %bit_mask = vector.constant 16843009 : vector<8xi32>
  %four = vector.constant 4 : vector<8xi32>
  %header_view = buffer.view %w[%origin] : buffer -> view<4xi32>
  %header = vector.load %header_view[0] : view<4xi32> -> vector<4xi32>
  %scale, %minimum = func.call @g_k_metadata(%header, %group) : (vector<4xi32>, index) -> (f32, f32)
  %page = index.div %group, %c2 : index
  %nibble = index.rem %group, %c2 : index
  %page_add = index.scale %page, %page_bytes : index, offset -> offset
  %payload = index.add %origin, %low_bytes : offset
  %codes_origin = index.add %payload, %page_add : offset
  %code_view = buffer.view %w[%codes_origin] : buffer -> view<8xi32>
  %packed = vector.load %code_view[0] : view<8xi32> -> vector<8xi32>
  %high_origin = index.add %origin, %high_bytes : offset
  %high_view = buffer.view %w[%high_origin] : buffer -> view<8xi32>
  %high_packed = vector.load %high_view[0] : view<8xi32> -> vector<8xi32>
  %shift_bits = index.mul %nibble, %c4 : index
  %shift_i32 = index.cast %shift_bits : index to i32
  %shift = vector.splat %shift_i32 : vector<8xi32>
  %shifted = vector.shrui %packed, %shift : vector<8xi32>
  %low = vector.andi %shifted, %nibble_mask : vector<8xi32>
  %group_i32 = index.cast %group : index to i32
  %group_shift = vector.splat %group_i32 : vector<8xi32>
  %high_shifted = vector.shrui %high_packed, %group_shift : vector<8xi32>
  %high_bit = vector.andi %high_shifted, %bit_mask : vector<8xi32>
  %high = vector.shli %high_bit, %four : vector<8xi32>
  %combined = vector.ori %low, %high : vector<8xi32>
  %bytes = vector.bitcast %combined : vector<8xi32> to vector<32xi8>
  %codes = vector.uitofp %bytes : vector<32xi8> to vector<32xf32>
  %scales = vector.splat %scale : vector<32xf32>
  %minima = vector.splat %minimum : vector<32xf32>
  %scaled = vector.mulf %codes, %scales : vector<32xf32>
  %values = vector.subf %scaled, %minima : vector<32xf32>
  func.return %values : vector<32xf32>
}

// Group `group` of a Q4_K superblock as raw codes plus (d*scale, dmin*min):
// value = codes * scale - minimum.
func.def inline @g_q4_k_codes(%w: buffer, %origin: offset, %group: index) -> (vector<32xf32>, f32, f32) {
  %c2 = index.constant 2 : index
  %c4 = index.constant 4 : index
  %header_bytes = index.constant 16 : offset
  %page_bytes = index.constant 32 : offset
  %nibble_mask = vector.constant 252645135 : vector<8xi32>
  %header_view = buffer.view %w[%origin] : buffer -> view<4xi32>
  %header = vector.load %header_view[0] : view<4xi32> -> vector<4xi32>
  %scale, %minimum = func.call @g_k_metadata(%header, %group) : (vector<4xi32>, index) -> (f32, f32)
  %page = index.div %group, %c2 : index
  %nibble = index.rem %group, %c2 : index
  %page_add = index.scale %page, %page_bytes : index, offset -> offset
  %payload = index.add %origin, %header_bytes : offset
  %codes_origin = index.add %payload, %page_add : offset
  %code_view = buffer.view %w[%codes_origin] : buffer -> view<8xi32>
  %packed = vector.load %code_view[0] : view<8xi32> -> vector<8xi32>
  %shift_bits = index.mul %nibble, %c4 : index
  %shift_i32 = index.cast %shift_bits : index to i32
  %shift = vector.splat %shift_i32 : vector<8xi32>
  %shifted = vector.shrui %packed, %shift : vector<8xi32>
  %masked = vector.andi %shifted, %nibble_mask : vector<8xi32>
  %bytes = vector.bitcast %masked : vector<8xi32> to vector<32xi8>
  %codes = vector.uitofp %bytes : vector<32xi8> to vector<32xf32>
  func.return %codes, %scale, %minimum : vector<32xf32>, f32, f32
}

func.def inline @g_store_row(%y: buffer, %tokens: index, %n: index, %token: index, %row: index, %value: f32, %accumulate: i32) {
  %zero = index.constant 0 : offset
  %c0_i32 = scalar.constant 0 : i32
  %bounded_token = index.assume %token [lt(%token, %tokens)] : index
  %bounded_row = index.assume %row [lt(%row, %n)] : index
  %y_view = buffer.view %y[%zero] : buffer -> view<[%tokens]x[%n]xf32>
  %add = scalar.cmpi ne, %accumulate, %c0_i32 : i32
  %result = scf.if %add -> (f32) {
    %old = view.load %y_view[%bounded_token, %bounded_row] : view<[%tokens]x[%n]xf32> -> f32
    %sum = scalar.addf %old, %value : f32
    scf.yield %sum : f32
  } else {
    scf.yield %value : f32
  }
  view.store %result, %y_view[%bounded_token, %bounded_row] : f32, view<[%tokens]x[%n]xf32>
  func.return
}
"""


COMMON = (COMMON.replace("%position_bytes = index.constant 512 : offset", f"%position_bytes = index.constant {POS_BYTES} : offset")
          .replace("// Paged KV: 104 planes (layer-major, K then V); one position holds 2 KV heads\n// x 128 F16 values (512 bytes)",
                   f"// Paged KV: {2 * L} planes (layer-major, K then V); one position holds {KVH} KV heads\n// x 128 F16 values ({POS_BYTES} bytes)"))

COMMON += r"""
// Q8_0 block (34 bytes): F16 scale d, 32 int8 codes; value = d * q.
func.def inline @g_q8_0_values(%w: buffer, %origin: offset, %group: index) -> (vector<32xf32>) {
  %two = index.constant 2 : offset
  %scale_view = buffer.view %w[%origin] : buffer -> view<1xf16>
  %code_origin = index.add %origin, %two : offset
  %code_view = buffer.view %w[%code_origin] : buffer -> view<32xi8>
  %scale_half = view.load %scale_view[0] : view<1xf16> -> f16
  %scale = scalar.extf %scale_half : f16 to f32
  %codes = vector.load %code_view[0] : view<32xi8> -> vector<32xi8>
  %values = vector.sitofp %codes : vector<32xi8> to vector<32xf32>
  %scales = vector.splat %scale : vector<32xf32>
  %scaled = vector.mulf %values, %scales : vector<32xf32>
  func.return %scaled : vector<32xf32>
}

func.def inline @g_q8_0_dot(%w: buffer, %origin: offset, %group: index, %inputs: vector<32xf32>) -> (f32) {
  %zero_f = scalar.constant 0.0 : f32
  %values = func.call @g_q8_0_values(%w, %origin, %group) : (buffer, offset, index) -> (vector<32xf32>)
  %dot = vector.dotf<reassoc|contract> %values, %inputs, %zero_f : vector<32xf32>, vector<32xf32>, f32
  func.return %dot : f32
}

// Q5_K group as values (for heads/embeddings) is g_q5_k_values; Q4_K is g_q4_k_values.
"""


def row_dot(fmt):
    """Partial dot of one lane's 32-value groups of a K-quant row with x[token]."""
    return f"""
func.def inline @g_{fmt}_row_dot(%w: buffer, %row_origin: offset, %x_view: view<[%tokens]x[%groups]x32xf32>, %tokens: index, %groups: index, %token: index, %lane: index, %lanes: index) -> (f32) {{
  %c0 = index.constant 0 : index
  %cgpb = index.constant {SB[fmt] // 32} : index
  %block_bytes = index.constant {FMT[fmt]} : offset
  %zero_f = scalar.constant 0.0 : f32
  %bounded_token = index.assume %token [lt(%token, %tokens)] : index
  %partial = scf.for %group = [%lane to %groups step %lanes](%sum = %zero_f : f32) -> (f32) {{
    %block = index.div %group, %cgpb : index
    %within = index.rem %group, %cgpb : index
    %block_add = index.scale %block, %block_bytes : index, offset -> offset
    %block_origin = index.add %row_origin, %block_add : offset
    %inputs = vector.load %x_view[%bounded_token, %group, %c0] : view<[%tokens]x[%groups]x32xf32> -> vector<32xf32>
    %dot = func.call @g_{fmt}_dot(%w, %block_origin, %within, %inputs) : (buffer, offset, index, vector<32xf32>) -> (f32)
    %next = scalar.addf %sum, %dot : f32
    scf.yield %next : f32
  }}
  func.return %partial : f32
}}
"""


def matmul(fmt, lanes=32):
    """y[t, n] (+)= W[n, :] . x[t, :]; one workgroup of `lanes` per (row, token)."""
    name = f"g_matmul_{fmt}" + ("" if lanes == 32 else f"_w{lanes}")
    return f"""
kernel.def export("{name}") @{name}(%n: index, %tokens: index) {{
  %c1 = index.constant 1 : index
  %c32 = index.constant {lanes} : index
  kernel.launch.config workgroups(%tokens, %n, %c1) workgroup_size(%c32, %c1, %c1) : index
}} launch(%k_arg: index, %n_arg: index, %tokens_arg: index, %accumulate: i32, %x: buffer, %w: buffer, %y: buffer) {{
  %k = index.assume %k_arg [range(%k_arg, 256, 65536), multiple_of(%k_arg, 256)] : index
  %n = index.assume %n_arg [range(%n_arg, 1, 262144)] : index
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %row_id = kernel.workgroup.id<y> : index
  %token_id = kernel.workgroup.id<x> : index
  %row = index.assume %row_id [lt(%row_id, %n)] : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %lane = kernel.workitem.id<x> : index
  %lanes = kernel.workgroup.size<x> : index
  %c0 = index.constant 0 : index
  %c32 = index.constant 32 : index
  %c256 = index.constant 256 : index
  %zero = index.constant 0 : offset
  %block_bytes = index.constant {FMT[fmt]} : offset
  %groups = index.div %k, %c32 : index
  %csb = index.constant {SB[fmt]} : index
  %blocks = index.div %k, %csb : index
  %row_bytes = index.scale %blocks, %block_bytes : index, offset -> offset
  %row_origin = index.scale %row, %row_bytes : index, offset -> offset
  %x_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x[%groups]x32xf32>
  %partial = func.call @g_{fmt}_row_dot(%w, %row_origin, %x_view, %tokens, %groups, %token, %lane, %lanes) : (buffer, offset, view<[%tokens]x[%groups]x32xf32>, index, index, index, index, index) -> (f32)
  %total = kernel.workgroup.reduce<addf> %partial : f32
  %first = index.cmp eq, %lane, %c0 : index
  scf.if %first {{
    func.call @g_store_row(%y, %tokens, %n, %token, %row, %total, %accumulate) : (buffer, index, index, index, index, f32, i32)
  }}
  kernel.return
}}
"""


def mv(fmt, K, R=16, TG=1, swiglu=False, suffix=""):
    """Decode matrix-vector: a workgroup covers one K row with one lane per
    32-value group (lanes = groups rounded up to a wave), R output rows and TG
    tokens. Each lane keeps its x group for the TG tokens in registers and
    decodes each weight group once for all of them, so weights stream once per
    token group and activations once per workgroup. Waves reduce with shuffles;
    one barrier folds the wave partials."""
    groups = K // 32
    lanes = (groups + 31) // 32 * 32
    waves = lanes // 32
    rb = row_bytes(fmt, K)
    nw = 2 if swiglu else 1
    name = f"g_mv_{fmt}_k{K}" + ("_swiglu" if swiglu else "") + (f"_t{TG}" if TG > 1 else "") + suffix
    wargs = "%w: buffer, %w2: buffer" if swiglu else "%w: buffer"
    L = [f"""
kernel.def export("{name}") @{name}(%n: index, %tokens: index) {{
  %c1 = index.constant 1 : index
  %lanes = index.constant {lanes} : index
  %rows_per = index.constant {R} : index
  %tg = index.constant {TG} : index
  %tg_less = index.constant {TG - 1} : index
  %row_groups = index.div %n, %rows_per : index
  %tg_pad = index.add %tokens, %tg_less : index
  %token_groups = index.div %tg_pad, %tg : index
  kernel.launch.config workgroups(%token_groups, %row_groups, %c1) workgroup_size(%lanes, %c1, %c1) : index
}} launch(%n_arg: index, %tokens_arg: index, %accumulate: i32, %x: buffer, {wargs}, %y: buffer) {{
  %n = index.assume %n_arg [range(%n_arg, {R}, 262144), multiple_of(%n_arg, {R})] : index
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %group_id = kernel.workgroup.id<y> : index
  %tgroup_id = kernel.workgroup.id<x> : index
  %workitem_id = kernel.workitem.id<x> : index
  %workitem = index.assume %workitem_id [range(%workitem_id, 0, {lanes - 1})] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, {waves - 1})] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c8 = index.constant 8 : index
  %rows_per = index.constant {R} : index
  %tg = index.constant {TG} : index
  %cW = index.constant {waves} : index
  %groups = index.constant {groups} : index
  %row_bytes = index.constant {rb} : offset
  %block_bytes = index.constant {FMT[fmt]} : offset
  %zero_f = scalar.constant 0.0 : f32
  %zero32 = vector.constant 0.0 : vector<32xf32>
  %x_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x{groups}x32xf32>
  %row_base = index.mul %group_id, %rows_per : index
  %token0 = index.mul %tgroup_id, %tg : index
  %ok = index.cmp ult, %workitem, %groups : index
  %g_last = index.constant {groups - 1} : index
  %g_c = index.min %workitem, %g_last : index
  %g = index.assume %g_c [range(%g_c, 0, {groups - 1})] : index
  %cgpb = index.constant {SB[fmt] // 32} : index
  %blk = index.div %g, %cgpb : index
  %within = index.rem %g, %cgpb : index
  %ba = index.scale %blk, %block_bytes : index, offset -> offset"""]
    for j in range(TG):
        L.append(f"""  %tj{j}_off = index.constant {j} : index
  %tok{j}_raw = index.add %token0, %tj{j}_off : index
  %live{j} = index.cmp ult, %tok{j}_raw, %tokens : index
  %use{j} = scalar.andi %live{j}, %ok : i1
  %x{j} = scf.if %use{j} -> (vector<32xf32>) {{
    %tk{j} = index.assume %tok{j}_raw [lt(%tok{j}_raw, %tokens)] : index
    %v = vector.load %x_view[%tk{j}, %g, %c0] : view<[%tokens]x{groups}x32xf32> -> vector<32xf32>
    scf.yield %v : vector<32xf32>
  }} else {{
    scf.yield %zero32 : vector<32xf32>
  }}
  %xs{j} = vector.reduce<addf> %x{j}, %zero_f : vector<32xf32>, f32""")
    L.append(f"""  %part_bytes = index.constant {waves * R * TG * nw * 4} : offset
  %parts = buffer.alloca<workgroup> align(16) %part_bytes : buffer
  %pv = buffer.view %parts[%zero] : buffer -> view<{nw}x{waves}x{R}x{TG}xf32>
  %first = index.cmp eq, %lane, %c0 : index
  scf.for %r = [%c0 to %rows_per step %c1] {{
    %row_value = index.add %row_base, %r : index
    %row = index.assume %row_value [lt(%row_value, %n)] : index
    %ro = index.scale %row, %row_bytes : index, offset -> offset
    %rb = index.assume %r [range(%r, 0, {R - 1})] : index
    %o = index.add %ro, %ba : offset""")
    for wi, w in enumerate(["w", "w2"] if swiglu else ["w"]):
        if fmt == "q4_k":
            # codes once per row; per token scale * dot(codes, x) - minimum * sum(x)
            L.append(f"""    %wv{wi}, %sc{wi}, %mn{wi} = scf.if %ok -> (vector<32xf32>, f32, f32) {{
      %v, %a, %b = func.call @g_q4_k_codes(%{w}, %o, %within) : (buffer, offset, index) -> (vector<32xf32>, f32, f32)
      scf.yield %v, %a, %b : vector<32xf32>, f32, f32
    }} else {{
      scf.yield %zero32, %zero_f, %zero_f : vector<32xf32>, f32, f32
    }}""")
        else:
            L.append(f"""    %wv{wi} = scf.if %ok -> (vector<32xf32>) {{
      %v = func.call @g_{fmt}_values(%{w}, %o, %within) : (buffer, offset, index) -> (vector<32xf32>)
      scf.yield %v : vector<32xf32>
    }} else {{
      scf.yield %zero32 : vector<32xf32>
    }}""")
        for j in range(TG):
            if fmt == "q4_k":
                L.append(f"""    %qd{wi}_{j} = vector.dotf<reassoc|contract> %wv{wi}, %x{j}, %zero_f : vector<32xf32>, vector<32xf32>, f32
    %qm{wi}_{j} = scalar.mulf %mn{wi}, %xs{j} : f32
    %pa{wi}_{j} = scalar.fmaf %qd{wi}_{j}, %sc{wi}, %qm{wi}_{j} : f32""".replace("scalar.fmaf %qd", "scalar.fmaf %qd") .replace(f"%pa{wi}_{j} = scalar.fmaf %qd{wi}_{j}, %sc{wi}, %qm{wi}_{j} : f32", f"%qs{wi}_{j} = scalar.mulf %qd{wi}_{j}, %sc{wi} : f32\n    %pa{wi}_{j} = scalar.subf %qs{wi}_{j}, %qm{wi}_{j} : f32") + f"""
    %s{wi}_{j} = kernel.subgroup.reduce<addf> %pa{wi}_{j} : f32
    scf.if %first {{
      view.store %s{wi}_{j}, %pv[{wi}, %wave, %rb, {j}] : f32, view<{nw}x{waves}x{R}x{TG}xf32>
    }}""")
                continue
            L.append(f"""    %pa{wi}_{j} = vector.dotf<reassoc|contract> %wv{wi}, %x{j}, %zero_f : vector<32xf32>, vector<32xf32>, f32
    %s{wi}_{j} = kernel.subgroup.reduce<addf> %pa{wi}_{j} : f32
    scf.if %first {{
      view.store %s{wi}_{j}, %pv[{wi}, %wave, %rb, {j}] : f32, view<{nw}x{waves}x{R}x{TG}xf32>
    }}""")
    L.append("  }")
    L.append(f"""  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %outputs = index.constant {R * TG} : index
  %lanes_step = index.constant {lanes} : index
  // R*TG outputs may exceed the lane count (one-wave rows): lanes stride over them.
  scf.for %wr0 = [%workitem to %outputs step %lanes_step] {{
    %wr = index.assume %wr0 [range(%wr0, 0, {R * TG - 1})] : index
    %rr_v = index.div %wr, %tg : index
    %rr = index.assume %rr_v [range(%rr_v, 0, {R - 1})] : index
    %jj_v = index.rem %wr, %tg : index
    %jj = index.assume %jj_v [range(%jj_v, 0, {TG - 1})] : index
    %tok_raw = index.add %token0, %jj : index
    %tok_live = index.cmp ult, %tok_raw, %tokens : index
    scf.if %tok_live {{""")
    for wi in range(nw):
        L.append(f"""      %t{wi} = scf.for %ww = [%c0 to %cW step %c1](%acc = %zero_f : f32) -> (f32) {{
        %bw = index.assume %ww [range(%ww, 0, {waves - 1})] : index
        %p = view.load %pv[{wi}, %bw, %rr, %jj] : view<{nw}x{waves}x{R}x{TG}xf32> -> f32
        %nx = scalar.addf %acc, %p : f32
        scf.yield %nx : f32
      }}""")
    if swiglu:
        L.append("""      %silu = scalar.siluf<afn> %t0 : f32
      %value = scalar.mulf %silu, %t1 : f32""")
    else:
        L.append("      %value = scalar.addf %t0, %zero_f : f32")
    L.append("""      %out_row = index.add %row_base, %rr : index
      func.call @g_store_row(%y, %tokens, %n, %tok_raw, %out_row, %value, %accumulate) : (buffer, index, index, index, index, f32, i32)
    }
  }
  kernel.return
}
""")
    return "\n".join(L)


EFMT = fmt_of("token_embd.weight")
_EN = """  %squares = vector.mulf %values, %values : vector<32xf32>
  %local = vector.reduce<addf> %squares, %zero_f : vector<32xf32>, f32
  %sum = kernel.workgroup.reduce<addf> %local : f32
  %eps0 = scalar.constant 1.0e-5 : f32
  %count0 = scalar.constant {H}.0 : f32
  %mean = scalar.divf %sum, %count0 : f32
  %stable = scalar.addf %mean, %eps0 : f32
  %inverse = scalar.rsqrtf %stable : f32
  %inverses = vector.splat %inverse : vector<32xf32>
  %normalized = vector.mulf %values, %inverses : vector<32xf32>
""".replace("{H}", str(H))
_ES = CFG.get("embed_scale")  # e.g. sqrt(hidden) for Gemma
EMBED_PRE = (_EN if EMBED_NORM else
             (f"  %escale = vector.splat {'%'}escale_s : vector<32xf32>\n  %normalized = vector.mulf %values, %escale : vector<32xf32>\n" if _ES else
              "  %normalized = vector.addf %values, %zero32 : vector<32xf32>\n"))
if _ES:
    EMBED_PRE = f"  %escale_s = scalar.constant {float(_ES)} : f32\n" + EMBED_PRE
EMBED_PRE += "  %count = scalar.constant " + str(float(H)) + " : f32\n  %eps = scalar.constant " + str(EPS) + " : f32\n"

NORMS = f"""
kernel.def export("g_token_info") @g_token_info(%tokens: index) {{
  %c1 = index.constant 1 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c1, %c1, %c1) : index
}} launch(%tokens_arg: index, %spans_arg: index, %metadata: buffer, %info: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %zero = index.constant 0 : offset
  %c16 = index.constant 16 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c0_i32 = scalar.constant 0 : i32
  %c1_i32 = scalar.constant 1 : i32
  %header = buffer.view %metadata[%zero] : buffer -> view<4xi32>
  %count32 = view.load %header[0] : view<4xi32> -> i32
  %count_bound = scalar.assume %count32 [range(%count32, 0, 8192)] : i32
  %count_value = index.cast %count_bound : i32 to index
  %fewer = index.cmp ult, %count_value, %spans : index
  %count = scf.select %fewer, %count_value, %spans : index
  %packets = buffer.view %metadata[%c16] : buffer -> view<[%spans]x9xi32>
  %token32 = index.cast %token : index to i32
  %minus_one = scalar.constant -1 : i32
  %row, %position, %live, %span = scf.for %s = [%c0 to %count step %c1](%r = %minus_one : i32, %p = %c0_i32 : i32, %l = %c0_i32 : i32, %found = %minus_one : i32) -> (i32, i32, i32, i32) {{
    %bounded_s = index.assume %s [lt(%s, %spans)] : index
    %length32 = view.load %packets[%bounded_s, 0] : view<[%spans]x9xi32> -> i32
    %position32 = view.load %packets[%bounded_s, 1] : view<[%spans]x9xi32> -> i32
    %row32 = view.load %packets[%bounded_s, 3] : view<[%spans]x9xi32> -> i32
    %input32 = view.load %packets[%bounded_s, 4] : view<[%spans]x9xi32> -> i32
    %end32 = scalar.addi %input32, %length32 : i32
    %after_begin = scalar.cmpi sge, %token32, %input32 : i32
    %before_end = scalar.cmpi slt, %token32, %end32 : i32
    %inside = scalar.andi %after_begin, %before_end : i1
    %relative = scalar.subi %token32, %input32 : i32
    %absolute = scalar.addi %position32, %relative : i32
    %s32 = index.cast %bounded_s : index to i32
    %next_r = scf.select %inside, %row32, %r : i32
    %next_p = scf.select %inside, %absolute, %p : i32
    %next_l = scf.select %inside, %c1_i32, %l : i32
    %next_s = scf.select %inside, %s32, %found : i32
    scf.yield %next_r, %next_p, %next_l, %next_s : i32, i32, i32, i32
  }}
  %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  view.store %row, %info_view[%token, 0] : i32, view<[%tokens]x4xi32>
  view.store %position, %info_view[%token, 1] : i32, view<[%tokens]x4xi32>
  view.store %live, %info_view[%token, 2] : i32, view<[%tokens]x4xi32>
  view.store %span, %info_view[%token, 3] : i32, view<[%tokens]x4xi32>
  kernel.return
}}

// x[t] = embedding[id] (optionally rmsnorm'd / scaled), h[t] = rmsnorm(x[t]) * w.
// 256 lanes; lane g < {HG} dequantizes {EFMT} group g of the row.
kernel.def export("g_embed") @g_embed(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %ids: buffer, %table: buffer, %w: buffer, %x: buffer, %h: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c8 = index.constant {SB[EFMT] // 32} : index
  %groups = index.constant {HG} : index
  %zero_f = scalar.constant 0.0 : f32
  %ids_view = buffer.view %ids[%zero] : buffer -> view<[%tokens]xi32>
  %raw = view.load %ids_view[%token] : view<[%tokens]xi32> -> i32
  %c0_i32 = scalar.constant 0 : i32
  %vocab_last = scalar.constant {V - 1} : i32
  %low = scalar.maxsi %raw, %c0_i32 : i32
  %clamped = scalar.minsi %low, %vocab_last : i32
  %bounded = scalar.assume %clamped [range(%clamped, 0, {V - 1})] : i32
  %id = index.cast %bounded : i32 to index
  %row_bytes = index.constant {row_bytes(EFMT, H)} : offset
  %block_bytes = index.constant {FMT[EFMT]} : offset
  %row_origin = index.scale %id, %row_bytes : index, offset -> offset
  %active = index.cmp ult, %lane, %groups : index
  %zero32 = vector.constant 0.0 : vector<32xf32>
  %values = scf.if %active -> (vector<32xf32>) {{
    %block = index.div %lane, %c8 : index
    %within = index.rem %lane, %c8 : index
    %block_add = index.scale %block, %block_bytes : index, offset -> offset
    %origin = index.add %row_origin, %block_add : offset
    %v = func.call @g_{EFMT}_values(%table, %origin, %within) : (buffer, offset, index) -> (vector<32xf32>)
    scf.yield %v : vector<32xf32>
  }} else {{
    scf.yield %zero32 : vector<32xf32>
  }}
{EMBED_PRE}  %squares2 = vector.mulf %normalized, %normalized : vector<32xf32>
  %local2 = vector.reduce<addf> %squares2, %zero_f : vector<32xf32>, f32
  %sum2 = kernel.workgroup.reduce<addf> %local2 : f32
  %mean2 = scalar.divf %sum2, %count : f32
  %stable2 = scalar.addf %mean2, %eps : f32
  %inverse2 = scalar.rsqrtf %stable2 : f32
  %inverses2 = vector.splat %inverse2 : vector<32xf32>
  scf.if %active {{
    %bl = index.assume %lane [range(%lane, 0, {HG - 1})] : index
    %x_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x{HG}x32xf32>
    %h_view = buffer.view %h[%zero] : buffer -> view<[%tokens]x{HG}x32xf32>
    %w_view = buffer.view %w[%zero] : buffer -> view<{HG}x32xf32>
    vector.store %normalized, %x_view[%token, %bl, %c0] : vector<32xf32>, view<[%tokens]x{HG}x32xf32>
    %weights = vector.load %w_view[%bl, %c0] : view<{HG}x32xf32> -> vector<32xf32>
    %n2 = vector.mulf %normalized, %inverses2 : vector<32xf32>
    %hv = vector.mulf %n2, %weights : vector<32xf32>
    vector.store %hv, %h_view[%token, %bl, %c0] : vector<32xf32>, view<[%tokens]x{HG}x32xf32>
  }}
  kernel.return
}}

// x[t] += {'rmsnorm(y[t]) * w_post' if POST_NORM else 'y[t]'}; h[t] = rmsnorm(x[t]) * w_next.
kernel.def export("g_post_add_norm") @g_post_add_norm(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %y: buffer, %w_post: buffer, %x: buffer, %w_next: buffer, %h: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %groups = index.constant {HG} : index
  %zero_f = scalar.constant 0.0 : f32
  %zero32 = vector.constant 0.0 : vector<32xf32>
  %count = scalar.constant {float(H)} : f32
  %active = index.cmp ult, %lane, %groups : index
  %bl_raw = index.min %lane, %groups : index
  %y_view = buffer.view %y[%zero] : buffer -> view<[%tokens]x{HG}x32xf32>
  %x_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x{HG}x32xf32>
  %yv, %xv = scf.if %active -> (vector<32xf32>, vector<32xf32>) {{
    %bl = index.assume %lane [range(%lane, 0, {HG - 1})] : index
    %a = vector.load %y_view[%token, %bl, %c0] : view<[%tokens]x{HG}x32xf32> -> vector<32xf32>
    %b = vector.load %x_view[%token, %bl, %c0] : view<[%tokens]x{HG}x32xf32> -> vector<32xf32>
    scf.yield %a, %b : vector<32xf32>, vector<32xf32>
  }} else {{
    scf.yield %zero32, %zero32 : vector<32xf32>, vector<32xf32>
  }}
  %ysq = vector.mulf %yv, %yv : vector<32xf32>
  %ylocal = vector.reduce<addf> %ysq, %zero_f : vector<32xf32>, f32
  %ysum = kernel.workgroup.reduce<addf> %ylocal : f32
  %ymean = scalar.divf %ysum, %count : f32
  %post_eps = scalar.constant {POST_EPS} : f32
  %ystable = scalar.addf %ymean, %post_eps : f32
  %yinv = scalar.rsqrtf %ystable : f32
  %yinvs = vector.splat %yinv : vector<32xf32>
  %w_post_view = buffer.view %w_post[%zero] : buffer -> view<{HG}x32xf32>
  %w_next_view = buffer.view %w_next[%zero] : buffer -> view<{HG}x32xf32>
  %xn = scf.if %active -> (vector<32xf32>) {{
    %bl = index.assume %lane [range(%lane, 0, {HG - 1})] : index
    %wp = vector.load %w_post_view[%bl, %c0] : view<{HG}x32xf32> -> vector<32xf32>
{"    %yn = vector.mulf %yv, %yinvs : vector<32xf32>" if POST_NORM else "    %yn = vector.addf %yv, %zero32 : vector<32xf32>"}
{"    %r = vector.mulf %yn, %wp : vector<32xf32>" if POST_NORM else "    %r = vector.addf %yn, %zero32 : vector<32xf32>"}
    %sum = vector.addf %xv, %r : vector<32xf32>
    vector.store %sum, %x_view[%token, %bl, %c0] : vector<32xf32>, view<[%tokens]x{HG}x32xf32>
    scf.yield %sum : vector<32xf32>
  }} else {{
    scf.yield %zero32 : vector<32xf32>
  }}
  %xsq = vector.mulf %xn, %xn : vector<32xf32>
  %xlocal = vector.reduce<addf> %xsq, %zero_f : vector<32xf32>, f32
  %xsum = kernel.workgroup.reduce<addf> %xlocal : f32
  %xmean = scalar.divf %xsum, %count : f32
  %eps = scalar.constant {EPS} : f32
  %xstable = scalar.addf %xmean, %eps : f32
  %xinv = scalar.rsqrtf %xstable : f32
  %xinvs = vector.splat %xinv : vector<32xf32>
  scf.if %active {{
    %bl = index.assume %lane [range(%lane, 0, {HG - 1})] : index
    %wn = vector.load %w_next_view[%bl, %c0] : view<{HG}x32xf32> -> vector<32xf32>
    %xnn = vector.mulf %xn, %xinvs : vector<32xf32>
    %hv = vector.mulf %xnn, %wn : vector<32xf32>
    %h_view = buffer.view %h[%zero] : buffer -> view<[%tokens]x{HG}x32xf32>
    vector.store %hv, %h_view[%token, %bl, %c0] : vector<32xf32>, view<[%tokens]x{HG}x32xf32>
  }}
  kernel.return
}}
"""


ROW_SCALE = r"""

// Per-row power-of-two scale (as g_row_scale) plus the scaled row as F16, the
// GEMM A operand: x16[r] = f16(x[r] * scale[r]).
kernel.def export("g_row_scale_f16") @g_row_scale_f16(%rows: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%rows, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%rows_arg: index, %k_arg: index, %x: buffer, %scales: buffer, %x16: buffer) {
  %rows = index.assume %rows_arg [range(%rows_arg, 1, 2097152)] : index
  %k = index.assume %k_arg [range(%k_arg, 256, 65536), multiple_of(%k_arg, 256)] : index
  %row_id = kernel.workgroup.id<x> : index
  %row = index.assume %row_id [lt(%row_id, %rows)] : index
  %lane = kernel.workitem.id<x> : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %zero_f = scalar.constant 0.0 : f32
  %x_view = buffer.view %x[%zero] : buffer -> view<[%rows]x[%k]xf32>
  %h_view = buffer.view %x16[%zero] : buffer -> view<[%rows]x[%k]xf16>
  %local = scf.for %c = [%lane to %k step %c256](%m = %zero_f : f32) -> (f32) {
    %bc = index.assume %c [lt(%c, %k)] : index
    %v = view.load %x_view[%row, %bc] : view<[%rows]x[%k]xf32> -> f32
    %a = scalar.absf %v : f32
    %n = scalar.maxnumf %m, %a : f32
    scf.yield %n : f32
  }
  %maximum = kernel.workgroup.reduce<maxnumf> %local : f32
  %bits = scalar.bitcast %maximum : f32 to i32
  %c23 = scalar.constant 23 : i32
  %c255 = scalar.constant 255 : i32
  %shifted = scalar.shrui %bits, %c23 : i32
  %biased = scalar.andi %shifted, %c255 : i32
  %c127 = scalar.constant 127 : i32
  %c13 = scalar.constant 13 : i32
  %c0_i32 = scalar.constant 0 : i32
  %c100 = scalar.constant 100 : i32
  %unbiased = scalar.subi %biased, %c127 : i32
  %raw_e = scalar.subi %unbiased, %c13 : i32
  %positive = scalar.maxsi %raw_e, %c0_i32 : i32
  %e = scalar.minsi %positive, %c100 : i32
  %scale_exp = scalar.subi %c127, %e : i32
  %inverse_exp = scalar.addi %c127, %e : i32
  %scale_bits = scalar.shli %scale_exp, %c23 : i32
  %inverse_bits = scalar.shli %inverse_exp, %c23 : i32
  %scale = scalar.bitcast %scale_bits : i32 to f32
  %inverse = scalar.bitcast %inverse_bits : i32 to f32
  scf.for %c = [%lane to %k step %c256] {
    %bc = index.assume %c [lt(%c, %k)] : index
    %v = view.load %x_view[%row, %bc] : view<[%rows]x[%k]xf32> -> f32
    %sv = scalar.mulf %v, %scale : f32
    %hv = scalar.fptrunc %sv : f32 to f16
    view.store %hv, %h_view[%row, %bc] : f16, view<[%rows]x[%k]xf16>
  }
  %first = index.cmp eq, %lane, %c0 : index
  scf.if %first {
    %s_view = buffer.view %scales[%zero] : buffer -> view<[%rows]x2xf32>
    view.store %scale, %s_view[%row, %c0] : f32, view<[%rows]x2xf32>
    view.store %inverse, %s_view[%row, %c1] : f32, view<[%rows]x2xf32>
  }
  kernel.return
}
// Per-row power-of-two scaling so F16 GEMM operands cannot overflow: rows
// whose max |value| exceeds 2^14 get scale 2^-e (max * scale < 2^15), others
// 1. Writes (scale, 1/scale) per row; power-of-two scaling is exact.
kernel.def export("g_row_scale") @g_row_scale(%rows: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%rows, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%rows_arg: index, %k_arg: index, %x: buffer, %scales: buffer) {
  %rows = index.assume %rows_arg [range(%rows_arg, 1, 2097152)] : index
  %k = index.assume %k_arg [range(%k_arg, 256, 65536), multiple_of(%k_arg, 256)] : index
  %row_id = kernel.workgroup.id<x> : index
  %row = index.assume %row_id [lt(%row_id, %rows)] : index
  %lane = kernel.workitem.id<x> : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c256 = index.constant 256 : index
  %zero_f = scalar.constant 0.0 : f32
  %x_view = buffer.view %x[%zero] : buffer -> view<[%rows]x[%k]xf32>
  %local = scf.for %c = [%lane to %k step %c256](%m = %zero_f : f32) -> (f32) {
    %bc = index.assume %c [lt(%c, %k)] : index
    %v = view.load %x_view[%row, %bc] : view<[%rows]x[%k]xf32> -> f32
    %a = scalar.absf %v : f32
    %n = scalar.maxnumf %m, %a : f32
    scf.yield %n : f32
  }
  %maximum = kernel.workgroup.reduce<maxnumf> %local : f32
  %first = index.cmp eq, %lane, %c0 : index
  scf.if %first {
    %bits = scalar.bitcast %maximum : f32 to i32
    %c23 = scalar.constant 23 : i32
    %c255 = scalar.constant 255 : i32
    %shifted = scalar.shrui %bits, %c23 : i32
    %biased = scalar.andi %shifted, %c255 : i32
    // e = exponent(max) - 13 when max >= 2^14 (biased exponent 141), else 0.
    %c127 = scalar.constant 127 : i32
    %c13 = scalar.constant 13 : i32
    %c0_i32 = scalar.constant 0 : i32
    %c100 = scalar.constant 100 : i32
    %unbiased = scalar.subi %biased, %c127 : i32
    %raw_e = scalar.subi %unbiased, %c13 : i32
    %positive = scalar.maxsi %raw_e, %c0_i32 : i32
    %e = scalar.minsi %positive, %c100 : i32
    %scale_exp = scalar.subi %c127, %e : i32
    %inverse_exp = scalar.addi %c127, %e : i32
    %scale_bits = scalar.shli %scale_exp, %c23 : i32
    %inverse_bits = scalar.shli %inverse_exp, %c23 : i32
    %scale = scalar.bitcast %scale_bits : i32 to f32
    %inverse = scalar.bitcast %inverse_bits : i32 to f32
    %s_view = buffer.view %scales[%zero] : buffer -> view<[%rows]x2xf32>
    view.store %scale, %s_view[%row, %c0] : f32, view<[%rows]x2xf32>
    %c1 = index.constant 1 : index
    view.store %inverse, %s_view[%row, %c1] : f32, view<[%rows]x2xf32>
  }
  kernel.return
}
"""


def qk_rope():
    """Per-head (optional) RMSNorm + RoPE for QH query and KVH key heads, with
    keys/values stored to the paged F16 KV planes: one 256-lane workgroup per
    token. Lane l of a wave owns 4 of a head's 128 dims (NeoX: l, l+32, l+64,
    l+96 with rotary pairs (0,2),(1,3); interleaved: 4l..4l+3 with pairs
    (0,1),(2,3)); its two angles are computed once and reused for every head."""
    if ROPE == "neox":
        dims = ["%lane", "%d1", "%d2", "%d3"]
        dim_defs = "  %d1 = index.add %lane, %c32 : index\n  %d2 = index.add %lane, %c64 : index\n  %d3 = index.add %lane, %c96 : index\n"
        freq_idx = ["%lane", "%d1"]; pairs = [(0, 2, 0), (1, 3, 1)]
    else:
        dims = ["%e0", "%e1", "%e2", "%e3"]
        dim_defs = "  %e0 = index.mul %lane, %c4 : index\n  %e1 = index.add %e0, %c1 : index\n  %e2 = index.add %e0, %c2 : index\n  %e3 = index.add %e0, %c3 : index\n  %f0 = index.mul %lane, %c2 : index\n  %f1 = index.add %f0, %c1 : index\n"
        freq_idx = ["%f0", "%f1"]; pairs = [(0, 1, 0), (2, 3, 1)]
    ln_theta_over_64 = math.log(THETA) / 64.0
    t = [f"""
kernel.def export("g_qk_rope_store") @g_qk_rope_store(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%tokens, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%tokens_arg: index, %layer_arg: index, %rope: i32, %context_arg: index, %pool_arg: index, %q: buffer, %k: buffer, %v: buffer, %q_norm: buffer, %k_norm: buffer, %info: buffer, %origins: buffer, %arena: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %layer = index.assume %layer_arg [range(%layer_arg, 0, {L - 1})] : index
  %context = index.assume %context_arg [range(%context_arg, 16, 262144)] : index
  %pool = index.assume %pool_arg [range(%pool_arg, 16, 1048576)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, 7)] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c2 = index.constant 2 : index
  %c3 = index.constant 3 : index
  %c4 = index.constant 4 : index
  %c8 = index.constant 8 : index
  %c15 = index.constant 15 : index
  %c16 = index.constant 16 : index
  %c32 = index.constant 32 : index
  %c64 = index.constant 64 : index
  %c96 = index.constant 96 : index
  %cqh = index.constant {QH} : index
  %call = index.constant {QH + KVH} : index
  %c0_i32 = scalar.constant 0 : i32
  %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %row32 = view.load %info_view[%token, 0] : view<[%tokens]x4xi32> -> i32
  %position32 = view.load %info_view[%token, 1] : view<[%tokens]x4xi32> -> i32
  %live32 = view.load %info_view[%token, 2] : view<[%tokens]x4xi32> -> i32
  %live = scalar.cmpi ne, %live32, %c0_i32 : i32
  %position_low = scalar.maxsi %position32, %c0_i32 : i32
{dim_defs}  %apply_rope = scalar.cmpi ne, %rope, %c0_i32 : i32
  %position_f = scalar.sitofp %position_low : i32 to f32
  %ln_theta_over_64 = scalar.constant {ln_theta_over_64!r} : f32
  %neg = scalar.constant -1.0 : f32
  %inv_two_pi = scalar.constant 0.15915494309189535 : f32
  %two_pi = scalar.constant 6.283185307179586 : f32
  %half = scalar.constant 0.5 : f32
  %one_f = scalar.constant 1.0 : f32
  %zero_f = scalar.constant 0.0 : f32
"""]
    for j, idx in enumerate(freq_idx):
        t.append(f"""  // angle = position * theta^(-2i/128) = position * exp(-i * ln(theta) / 64)
  %li{j} = index.cast {idx} : index to i32
  %lf{j} = scalar.sitofp %li{j} : i32 to f32
  %ex{j}a = scalar.mulf %lf{j}, %ln_theta_over_64 : f32
  %ex{j}b = scalar.mulf %ex{j}a, %neg : f32
  %if{j} = scalar.expf<afn> %ex{j}b : f32
  %ra{j} = scalar.mulf %position_f, %if{j} : f32
  %tr{j} = scalar.mulf %ra{j}, %inv_two_pi : f32
  %ts{j} = scalar.addf %tr{j}, %half : f32
  %ti{j} = scalar.fptosi %ts{j} : f32 to i32
  %tf{j} = scalar.sitofp %ti{j} : i32 to f32
  %wr{j} = scalar.mulf %tf{j}, %two_pi : f32
  %an{j} = scalar.subf %ra{j}, %wr{j} : f32
  %cr{j} = scalar.cosf<afn> %an{j} : f32
  %sr{j} = scalar.sinf<afn> %an{j} : f32
  %cos{j} = scf.select %apply_rope, %cr{j}, %one_f : f32
  %sin{j} = scf.select %apply_rope, %sr{j}, %zero_f : f32""")
    t.append(f"""  %q_view = buffer.view %q[%zero] : buffer -> view<[%tokens]x{QH}x128xf32>
  %k_view = buffer.view %k[%zero] : buffer -> view<[%tokens]x{KVH}x128xf32>
  %v_view = buffer.view %v[%zero] : buffer -> view<[%tokens]x{KVH}x128xf32>
  %qn_view = buffer.view %q_norm[%zero] : buffer -> view<128xf32>
  %kn_view = buffer.view %k_norm[%zero] : buffer -> view<128xf32>""")
    for i in range(4):
        t.append(f"  %w{i}q = view.load %qn_view[{dims[i]}] : view<128xf32> -> f32\n  %w{i}k = view.load %kn_view[{dims[i]}] : view<128xf32> -> f32")
    t.append(f"""  %dims = scalar.constant 128.0 : f32
  %eps = scalar.constant {EPS!r} : f32
  scf.for %head = [%wave to %call step %c8] {{
    %bh = index.assume %head [range(%head, 0, {QH + KVH - 1})] : index
    %is_query = index.cmp ult, %bh, %cqh : index
    %kvh_value = index.sub %bh, %cqh : index
    %a0, %a1, %a2, %a3 = scf.if %is_query -> (f32, f32, f32, f32) {{
      %qh = index.assume %bh [range(%bh, 0, {QH - 1})] : index""")
    for i in range(4):
        t.append(f"      %x{i} = view.load %q_view[%token, %qh, {dims[i]}] : view<[%tokens]x{QH}x128xf32> -> f32")
    t.append(f"""      scf.yield %x0, %x1, %x2, %x3 : f32, f32, f32, f32
    }} else {{
      %kh = index.assume %kvh_value [range(%kvh_value, 0, {KVH - 1})] : index""")
    for i in range(4):
        t.append(f"      %x{i} = view.load %k_view[%token, %kh, {dims[i]}] : view<[%tokens]x{KVH}x128xf32> -> f32")
    t.append("""      scf.yield %x0, %x1, %x2, %x3 : f32, f32, f32, f32
    }""")
    if QK_NORM:
        t.append("""    %s0 = scalar.mulf %a0, %a0 : f32
    %s1 = scalar.fmaf %a1, %a1, %s0 : f32
    %s2 = scalar.fmaf %a2, %a2, %s1 : f32
    %s3 = scalar.fmaf %a3, %a3, %s2 : f32
    %sum = kernel.subgroup.reduce<addf> %s3 : f32
    %mean = scalar.divf %sum, %dims : f32
    %stable = scalar.addf %mean, %eps : f32
    %inverse = scalar.rsqrtf %stable : f32""")
        for i in range(4):
            t.append(f"""    %w{i} = scf.select %is_query, %w{i}q, %w{i}k : f32
    %n{i}a = scalar.mulf %a{i}, %inverse : f32
    %n{i} = scalar.mulf %n{i}a, %w{i} : f32""")
    else:
        for i in range(4):
            t.append(f"    %n{i} = scalar.addf %a{i}, %zero_f : f32")
    for (x, y, j) in pairs:
        t.append(f"""    %rc{x} = scalar.mulf %n{x}, %cos{j} : f32
    %rs{y} = scalar.mulf %n{y}, %sin{j} : f32
    %r{x} = scalar.subf %rc{x}, %rs{y} : f32
    %rc{y} = scalar.mulf %n{y}, %cos{j} : f32
    %rs{x} = scalar.mulf %n{x}, %sin{j} : f32
    %r{y} = scalar.addf %rc{y}, %rs{x} : f32""")
    t.append(f"""    scf.if %is_query {{
      %qh = index.assume %bh [range(%bh, 0, {QH - 1})] : index""")
    for i in range(4):
        t.append(f"      view.store %r{i}, %q_view[%token, %qh, {dims[i]}] : f32, view<[%tokens]x{QH}x128xf32>")
    t.append(f"""    }} else {{
      scf.if %live {{
        %kh = index.assume %kvh_value [range(%kvh_value, 0, {KVH - 1})] : index
        %row_bound = scalar.assume %row32 [range(%row32, 0, 2147483646)] : i32
        %row = index.cast %row_bound : i32 to index
        %position_bound = scalar.assume %position_low [range(%position_low, 0, 262143)] : i32
        %position = index.cast %position_bound : i32 to index
        %padded = index.add %context, %c15 : index
        %blocks = index.div %padded, %c16 : index
        %k_plane = index.mul %layer, %c2 : index
        %k_offset = func.call @g_kv_offset(%origins, %row, %position, %blocks, %k_plane, %pool) : (buffer, index, index, index, index, index) -> (offset)
        %position_bytes_v = index.constant {POS_BYTES} : offset
        %plane_stride = index.scale %pool, %position_bytes_v : index, offset -> offset
        %v_offset = index.add %k_offset, %plane_stride : offset
        %k_cache = buffer.view %arena[%k_offset] : buffer -> view<{KVH}x128xf16>
        %v_cache = buffer.view %arena[%v_offset] : buffer -> view<{KVH}x128xf16>""")
    for i in range(4):
        t.append(f"""        %h{i} = scalar.fptrunc %r{i} : f32 to f16
        view.store %h{i}, %k_cache[%kh, {dims[i]}] : f16, view<{KVH}x128xf16>
        %vv{i} = view.load %v_view[%token, %kh, {dims[i]}] : view<[%tokens]x{KVH}x128xf32> -> f32
        %vh{i} = scalar.fptrunc %vv{i} : f32 to f16
        view.store %vh{i}, %v_cache[%kh, {dims[i]}] : f16, view<{KVH}x128xf16>""")
    t.append("""      }
    }
  }
  kernel.return
}
""")
    return "\n".join(t)


def listed(text, name, grid_rest):
    """Variant of a per-token kernel dispatched over the tokens in `list`
    (count read from a device buffer for indirect dispatch)."""
    a = text.index(f'kernel.def export("{name}")')
    b = text.index("launch(", a)
    header = f"""kernel.def export("{name}_listed") @{name}_listed(%count_buffer: buffer) {{
  %origin_c = index.constant 0 : offset
  %counts = buffer.view %count_buffer[%origin_c] : buffer -> view<1xi32>
  %word = view.load %counts[0] : view<1xi32> -> i32
  %bounded = scalar.assume %word [range(%word, 0, 8192)] : i32
  %count = index.cast %bounded : i32 to index
{grid_rest}
}} """
    body = text[b:]
    body = body.replace("launch(%tokens_arg: index, ", "launch(%tokens_arg: index, %list: buffer, ", 1)
    old_token = """  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index"""
    assert old_token in body
    body = body.replace(old_token, """  %slot = kernel.workgroup.id<x> : index
  %zero_l = index.constant 0 : offset
  %list_view = buffer.view %list[%zero_l] : buffer -> view<[%tokens]xi32>
  %bounded_slot = index.assume %slot [lt(%slot, %tokens)] : index
  %token32 = view.load %list_view[%bounded_slot] : view<[%tokens]xi32> -> i32
  %token_bound = scalar.assume %token32 [range(%token32, 0, 8191)] : i32
  %token_value = index.cast %token_bound : i32 to index
  %token = index.assume %token_value [lt(%token_value, %tokens)] : index""", 1)
    end = body.index("kernel.return\n}\n") + len("kernel.return\n}\n")
    return header + body[:end]


def attention_text(S, P):
    """Split-K causal GQA attention: workgroup (token, KV head, split) of 16
    waves, wave w = query head 16*kv + w, lane l = dims 4l..4l+3. Sliding
    layers see the last 2048 positions (including the current one). The
    combine kernel folds the splits and applies the sigmoid output gate."""
    U = UNROLL
    body = []
    for i in range(U):
        body.append(f"""      %key{i}_raw = index.add %j, %c{i}u : index
      %ok{i} = index.cmp ult, %key{i}_raw, %stop : index
      %key{i}_c = scf.select %ok{i}, %key{i}_raw, %j : index
      %kp{i} = index.add %window_low, %key{i}_c : index
      %ko{i} = func.call @g_kv_offset(%origins, %row, %kp{i}, %blocks, %k_plane, %pool) : (buffer, index, index, index, index, index) -> (offset)
      %kc{i} = buffer.view %arena[%ko{i}] : buffer -> view<{KVH}x32x4xf16>
      %kh{i} = vector.load %kc{i}[%kv_head, %lane, %c0] : view<{KVH}x32x4xf16> -> vector<4xf16>
      %k{i} = vector.extf %kh{i} : vector<4xf16> to vector<4xf32>
      %vo{i} = index.add %ko{i}, %plane_stride : offset
      %vc{i} = buffer.view %arena[%vo{i}] : buffer -> view<{KVH}x32x4xf16>
      %vh{i} = vector.load %vc{i}[%kv_head, %lane, %c0] : view<{KVH}x32x4xf16> -> vector<4xf16>
      %v{i} = vector.extf %vh{i} : vector<4xf16> to vector<4xf32>""")
    for i in range(U):
        body.append(f"""      %prod{i} = vector.mulf %q4, %k{i} : vector<4xf32>
      %part{i} = vector.reduce<addf> %prod{i}, %zero_f : vector<4xf32>, f32
      %sr{i} = kernel.subgroup.reduce<addf> %part{i} : f32
      %s{i} = scf.select %ok{i}, %sr{i}, %neg_big : f32""")
    prev = "%m"
    for i in range(U):
        body.append(f"      %mx{i} = scalar.maxnumf {prev}, %s{i} : f32"); prev = f"%mx{i}"
    body.append(f"""      %m_shift = scalar.subf %m, {prev} : f32
      %alpha = scalar.expf<afn> %m_shift : f32
      %l_scaled = scalar.mulf %l, %alpha : f32
      %alpha_v = vector.splat %alpha : vector<4xf32>
      %acc_s = vector.mulf %acc, %alpha_v : vector<4xf32>""")
    pl, pa = "%l_scaled", "%acc_s"
    for i in range(U):
        body.append(f"""      %d{i} = scalar.subf %s{i}, {prev} : f32
      %p{i} = scalar.expf<afn> %d{i} : f32
      %la{i} = scalar.addf {pl}, %p{i} : f32
      %pv{i} = vector.splat %p{i} : vector<4xf32>
      %aa{i} = vector.fmaf %pv{i}, %v{i}, {pa} : vector<4xf32>"""); pl, pa = f"%la{i}", f"%aa{i}"
    body.append(f"      scf.yield {prev}, {pl}, {pa} : f32, f32, vector<4xf32>")
    consts = "".join(f"  %c{i}u = index.constant {i} : index\n" for i in range(U))
    GATE_TEXT = """    %gate_view = buffer.view %gate[%zero] : buffer -> view<[%tokens]x{QH}x32x4xf32>
    %g = vector.load %gate_view[%token, %head, %lane, %c0] : view<[%tokens]x{QH}x32x4xf32> -> vector<4xf32>
    %sig = vector.logisticf<afn> %g : vector<4xf32>
    %result = vector.mulf %attended, %sig : vector<4xf32>""".replace("{QH}", str(QH)) if GATE else "    %result = vector.addf %attended, %zero4 : vector<4xf32>"
    text = f"""
// Splits for `visible` keys: about 32 keys each, at most {S}.
func.def inline @{P}_split_count(%visible: index) -> (index) {{
  %c1 = index.constant 1 : index
  %c127 = index.constant 31 : index
  %c128 = index.constant 32 : index
  %cmax = index.constant {S} : index
  %padded = index.add %visible, %c127 : index
  %target = index.div %padded, %c128 : index
  %capped = index.min %target, %cmax : index
  %count_value = index.max %capped, %c1 : index
  %count = index.assume %count_value [range(%count_value, 1, {S})] : index
  func.return %count : index
}}

kernel.def export("{P}_attention_split") @{P}_attention_split(%tokens: index) {{
  %c1 = index.constant 1 : index
  %ckvh = index.constant {KVH} : index
  %cS = index.constant {S} : index
  %clanes = index.constant {G * 32} : index
  kernel.launch.config workgroups(%tokens, %ckvh, %cS) workgroup_size(%clanes, %c1, %c1) : index
}} launch(%tokens_arg: index, %layer_arg: index, %sliding: i32, %context_arg: index, %pool_arg: index, %q: buffer, %info: buffer, %origins: buffer, %arena: buffer, %part_acc: buffer, %part_ml: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %layer = index.assume %layer_arg [range(%layer_arg, 0, 51)] : index
  %context = index.assume %context_arg [range(%context_arg, 16, 262144)] : index
  %pool = index.assume %pool_arg [range(%pool_arg, 16, 1048576)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %kv_id = kernel.workgroup.id<y> : index
  %kv_head = index.assume %kv_id [range(%kv_id, 0, {KVH - 1})] : index
  %split_id = kernel.workgroup.id<z> : index
  %split = index.assume %split_id [range(%split_id, 0, {S - 1})] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, {G - 1})] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c2 = index.constant 2 : index
  %c15 = index.constant 15 : index
  %c16 = index.constant 16 : index
  %window = index.constant {WINDOW or 262144} : index
  %window_less = index.constant {(WINDOW or 262144) - 1} : index
  %c0_i32 = scalar.constant 0 : i32
  %cU = index.constant {U} : index
{consts}  %zero_f = scalar.constant 0.0 : f32
  %neg_big = scalar.constant -3.0e38 : f32
  %cg = index.constant {G} : index
  %head = index.madd %kv_head, %cg, %wave : index
  %bounded_head = index.assume %head [range(%head, 0, {QH - 1})] : index
  %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %row32 = view.load %info_view[%token, 0] : view<[%tokens]x4xi32> -> i32
  %position32 = view.load %info_view[%token, 1] : view<[%tokens]x4xi32> -> i32
  %live32 = view.load %info_view[%token, 2] : view<[%tokens]x4xi32> -> i32
  %needed = scalar.constant true : i1
  %live_token = scalar.cmpi ne, %live32, %c0_i32 : i32
  %live = scalar.andi %live_token, %needed : i1
  scf.if %live {{
    %row_bound = scalar.assume %row32 [range(%row32, 0, 2147483646)] : i32
    %row = index.cast %row_bound : i32 to index
    %position_bound = scalar.assume %position32 [range(%position32, 0, 262143)] : i32
    %position = index.cast %position_bound : i32 to index
    %padded = index.add %context, %c15 : index
    %blocks = index.div %padded, %c16 : index
    %k_plane = index.mul %layer, %c2 : index
    %v_plane = index.add %k_plane, %c1 : index
    // V of a position sits one plane (pool positions x 512 bytes) after its K.
    %position_bytes_v = index.constant {POS_BYTES} : offset
    %plane_stride = index.scale %pool, %position_bytes_v : index, offset -> offset
    %is_sliding = scalar.cmpi ne, %sliding, %c0_i32 : i32
    %far = index.cmp uge, %position, %window : index
    %use_window = scalar.andi %is_sliding, %far : i1
    %window_start = index.sub %position, %window_less : index
    %window_low = scf.select %use_window, %window_start, %c0 : index
    %end = index.add %position, %c1 : index
    %visible_value = index.sub %end, %window_low : index
    %visible = index.assume %visible_value [range(%visible_value, 1, 262144)] : index
    %s_eff = func.call @{P}_split_count(%visible) : (index) -> (index)
    %active_split = index.cmp ult, %split, %s_eff : index
    scf.if %active_split {{
    %s_less = index.sub %s_eff, %c1 : index
    %per_padded = index.add %visible, %s_less : index
    %per = index.div %per_padded, %s_eff : index
    %start_raw = index.mul %split, %per : index
    %start = index.min %start_raw, %visible : index
    %stop_raw = index.add %start_raw, %per : index
    %stop = index.min %stop_raw, %visible : index
    %q_view = buffer.view %q[%zero] : buffer -> view<[%tokens]x{QH}x32x4xf32>
    %q_raw = vector.load %q_view[%token, %bounded_head, %lane, %c0] : view<[%tokens]x{QH}x32x4xf32> -> vector<4xf32>
    %scale = scalar.constant 0.08838834764831845 : f32
    %scale_v = vector.splat %scale : vector<4xf32>
    %q4 = vector.mulf %q_raw, %scale_v : vector<4xf32>
    %zero4 = vector.splat %zero_f : vector<4xf32>
    %m_final, %l_final, %acc_final = scf.for %j = [%start to %stop step %cU](%m = %neg_big : f32, %l = %zero_f : f32, %acc = %zero4 : vector<4xf32>) -> (f32, f32, vector<4xf32>) {{
{chr(10).join(body)}
    }}
    %acc_view = buffer.view %part_acc[%zero] : buffer -> view<[%tokens]x{QH}x{S}x32x4xf32>
    vector.store %acc_final, %acc_view[%token, %bounded_head, %split, %lane, %c0] : vector<4xf32>, view<[%tokens]x{QH}x{S}x32x4xf32>
    %first = index.cmp eq, %lane, %c0 : index
    scf.if %first {{
      %ml_view = buffer.view %part_ml[%zero] : buffer -> view<[%tokens]x{QH}x{S}x2xf32>
      view.store %m_final, %ml_view[%token, %bounded_head, %split, 0] : f32, view<[%tokens]x{QH}x{S}x2xf32>
      view.store %l_final, %ml_view[%token, %bounded_head, %split, 1] : f32, view<[%tokens]x{QH}x{S}x2xf32>
    }}
    }}
  }}
  kernel.return
}}

// o[t, head] = (sum_s e^(m_s - M) acc_s / sum_s e^(m_s - M) l_s) * sigmoid(gate[t, head]).
kernel.def export("{P}_attention_combine") @{P}_attention_combine(%tokens: index) {{
  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  %cqh = index.constant {QH} : index
  kernel.launch.config workgroups(%tokens, %cqh, %c1) workgroup_size(%c32, %c1, %c1) : index
}} launch(%tokens_arg: index, %sliding: i32, %info: buffer, %part_acc: buffer, %part_ml: buffer, %gate: buffer, %out: buffer) {{
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %token_id = kernel.workgroup.id<x> : index
  %token = index.assume %token_id [lt(%token_id, %tokens)] : index
  %head_id = kernel.workgroup.id<y> : index
  %head = index.assume %head_id [range(%head_id, 0, {QH - 1})] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %cSc = index.constant {S} : index
  %c0_i32 = scalar.constant 0 : i32
  %zero_f = scalar.constant 0.0 : f32
  %neg_big = scalar.constant -3.0e38 : f32
  %zero4 = vector.splat %zero_f : vector<4xf32>
  %info_view = buffer.view %info[%zero] : buffer -> view<[%tokens]x4xi32>
  %live32 = view.load %info_view[%token, 2] : view<[%tokens]x4xi32> -> i32
  %live = scalar.cmpi ne, %live32, %c0_i32 : i32
  %needed = scalar.constant true : i1
  %out_view = buffer.view %out[%zero] : buffer -> view<[%tokens]x{QH}x32x4xf32>
  scf.if %needed {{
  scf.if %live {{
    %position32 = view.load %info_view[%token, 1] : view<[%tokens]x4xi32> -> i32
    %position_bound = scalar.assume %position32 [range(%position32, 0, 262143)] : i32
    %position = index.cast %position_bound : i32 to index
    %window = index.constant {WINDOW or 262144} : index
    %window_less = index.constant {(WINDOW or 262144) - 1} : index
    %is_sliding = scalar.cmpi ne, %sliding, %c0_i32 : i32
    %far = index.cmp uge, %position, %window : index
    %use_window = scalar.andi %is_sliding, %far : i1
    %window_start = index.sub %position, %window_less : index
    %window_low = scf.select %use_window, %window_start, %c0 : index
    %end = index.add %position, %c1 : index
    %visible_value = index.sub %end, %window_low : index
    %visible = index.assume %visible_value [range(%visible_value, 1, 262144)] : index
    %s_eff = func.call @{P}_split_count(%visible) : (index) -> (index)
    %ml_view = buffer.view %part_ml[%zero] : buffer -> view<[%tokens]x{QH}x{S}x2xf32>
    %acc_view = buffer.view %part_acc[%zero] : buffer -> view<[%tokens]x{QH}x{S}x32x4xf32>
    %maximum = scf.for %s = [%c0 to %s_eff step %c1](%mx = %neg_big : f32) -> (f32) {{
      %bs = index.assume %s [range(%s, 0, {S - 1})] : index
      %ms = view.load %ml_view[%token, %head, %bs, 0] : view<[%tokens]x{QH}x{S}x2xf32> -> f32
      %next = scalar.maxnumf %mx, %ms : f32
      scf.yield %next : f32
    }}
    %den, %total = scf.for %s = [%c0 to %s_eff step %c1](%d = %zero_f : f32, %acc = %zero4 : vector<4xf32>) -> (f32, vector<4xf32>) {{
      %bs = index.assume %s [range(%s, 0, {S - 1})] : index
      %ms = view.load %ml_view[%token, %head, %bs, 0] : view<[%tokens]x{QH}x{S}x2xf32> -> f32
      %ls = view.load %ml_view[%token, %head, %bs, 1] : view<[%tokens]x{QH}x{S}x2xf32> -> f32
      %shift = scalar.subf %ms, %maximum : f32
      %w = scalar.expf<afn> %shift : f32
      %wl = scalar.mulf %w, %ls : f32
      %d_next = scalar.addf %d, %wl : f32
      %part = vector.load %acc_view[%token, %head, %bs, %lane, %c0] : view<[%tokens]x{QH}x{S}x32x4xf32> -> vector<4xf32>
      %w_v = vector.splat %w : vector<4xf32>
      %acc_next = vector.fmaf %w_v, %part, %acc : vector<4xf32>
      scf.yield %d_next, %acc_next : f32, vector<4xf32>
    }}
    %one = scalar.constant 1.0 : f32
    %inv = scalar.divf %one, %den : f32
    %inv_v = vector.splat %inv : vector<4xf32>
    %attended = vector.mulf %total, %inv_v : vector<4xf32>
{GATE_TEXT}
    vector.store %result, %out_view[%token, %head, %lane, %c0] : vector<4xf32>, view<[%tokens]x{QH}x32x4xf32>
  }} else {{
    vector.store %zero4, %out_view[%token, %head, %lane, %c0] : vector<4xf32>, view<[%tokens]x{QH}x32x4xf32>
  }}
  }}
  kernel.return
}}
"""
    split_l = listed(text, f"{P}_attention_split", f"""  %c1 = index.constant 1 : index
  %ckvh = index.constant {KVH} : index
  %cS = index.constant {S} : index
  %clanes = index.constant {G * 32} : index
  kernel.launch.config workgroups(%count, %ckvh, %cS) workgroup_size(%clanes, %c1, %c1) : index""")
    combine_l = listed(text, f"{P}_attention_combine", f"""  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  %cqh = index.constant {QH} : index
  kernel.launch.config workgroups(%count, %cqh, %c1) workgroup_size(%c32, %c1, %c1) : index""")
    return text, split_l, combine_l


def attention():
    """Decode attention with up to 64 splits; the prefill fallback (listed,
    indexed by token over up to 1024 rows) keeps 16 splits."""
    dec, _, _ = attention_text(64, "g")
    pre, split_l, combine_l = attention_text(16, "g16")
    count16 = pre[pre.index("// Splits for"):pre.index('kernel.def export("g16_attention_split")')]
    return dec + "\n" + count16 + split_l + "\n" + combine_l + FALLBACK


FALLBACK = r"""
// Tokens of 16-token tiles that g_attention_flash does not cover (mixed rows,
// non-consecutive positions or padding), listed once per epoch; count[0] is
// the list length for indirect dispatch.
kernel.def export("g_attention_fallback_list") @g_attention_fallback_list(%tokens: index) {
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%c1, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
} launch(%tokens_arg: index, %info: buffer, %list: buffer, %count: buffer) {
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 16, 8192), multiple_of(%tokens_arg, 16)] : index
  %lane_id = kernel.workitem.id<x> : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 255)] : index
  %zero = index.constant 0 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c16 = index.constant 16 : index
  %c256 = index.constant 256 : index
  %c0_i32 = scalar.constant 0 : i32
  %c16_i32 = scalar.constant 16 : i32
  %counter_bytes = index.constant 16 : offset
  %counter = buffer.alloca<workgroup> align(16) %counter_bytes : buffer
  %counter_view = buffer.view %counter[%zero] : buffer -> view<4xi32>
  %first = index.cmp eq, %lane, %c0 : index
  scf.if %first {
    view.store %c0_i32, %counter_view[%c0] : i32, view<4xi32>
  }
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  %tiles = index.div %tokens, %c16 : index
  %list_view = buffer.view %list[%zero] : buffer -> view<[%tokens]xi32>
  scf.for %tile = [%lane to %tiles step %c256] {
    %tile_first = index.mul %tile, %c16 : index
    %uniform, %r_unused, %p_unused = func.call @g_tile_uniform(%info, %tokens, %tile_first) : (buffer, index, index) -> (i1, i32, i32)
    %true = scalar.constant true : i1
    %not_uniform = scalar.xori %uniform, %true : i1
    scf.if %not_uniform {
      %base32 = view.atomic.rmw<addi> %c16_i32, %counter_view[%c0] {ordering = relaxed, scope = workgroup} : i32, view<4xi32> -> i32
      %base_bound = scalar.assume %base32 [range(%base32, 0, 8176)] : i32
      %base = index.cast %base_bound : i32 to index
      scf.for %i = [%c0 to %c16 step %c1] {
        %slot_value = index.add %base, %i : index
        %slot = index.assume %slot_value [lt(%slot_value, %tokens)] : index
        %token_value = index.add %tile_first, %i : index
        %token32 = index.cast %token_value : index to i32
        view.store %token32, %list_view[%slot] : i32, view<[%tokens]xi32>
      }
    }
  }
  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)
  scf.if %first {
    %total = view.load %counter_view[%c0] : view<4xi32> -> i32
    %count_view = buffer.view %count[%zero] : buffer -> view<1xi32>
    view.store %total, %count_view[0] : i32, view<1xi32>
  }
  kernel.return
}
"""


CHUNKS = (V + 127) // 128


def eos_text():
    ids = CFG.get("extra_eos_ids", [])
    t = ["        %terminal_0 = scalar.constant false : i1"]
    for i, e in enumerate(ids):
        t.append(f"        %eid{i} = scalar.constant {e} : i32\n        %eis{i} = scalar.cmpi eq, %winner, %eid{i} : i32\n        %terminal_{i + 1} = scalar.ori %terminal_{i}, %eis{i} : i1")
    t.append(f"        %token = scf.select %terminal_{len(ids)}, %eos, %winner : i32")
    return "\n".join(t)


def head():
    """Final argmax over the per-chunk (max logit, argmax) partials of each
    selected span; extra end-of-turn ids map to the service terminal."""
    return f"""
kernel.def export("g_head_select") @g_head_select(%spans: index) {{
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%spans, %c1, %c1) workgroup_size(%c256, %c1, %c1) : index
}} launch(%spans_arg: index, %metadata: buffer, %partials: buffer, %outputs: buffer) {{
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %span_id = kernel.workgroup.id<x> : index
  %span = index.assume %span_id [lt(%span_id, %spans)] : index
  %lane = kernel.workitem.id<x> : index
  %zero = index.constant 0 : offset
  %c16 = index.constant 16 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c256 = index.constant 256 : index
  %chunks = index.constant {CHUNKS} : index
  %c0_i32 = scalar.constant 0 : i32
  %c1_i32 = scalar.constant 1 : i32
  %header = buffer.view %metadata[%zero] : buffer -> view<4xi32>
  %count32 = view.load %header[0] : view<4xi32> -> i32
  %count_bound = scalar.assume %count32 [range(%count32, 0, 8192)] : i32
  %count = index.cast %count_bound : i32 to index
  %live_span = index.cmp ult, %span, %count : index
  scf.if %live_span {{
    %packets = buffer.view %metadata[%c16] : buffer -> view<[%spans]x9xi32>
    %flags = view.load %packets[%span, 6] : view<[%spans]x9xi32> -> i32
    %bits = scalar.andi %flags, %c1_i32 : i32
    %selected = scalar.cmpi ne, %bits, %c0_i32 : i32
    scf.if %selected {{
      %partial_view = buffer.view %partials[%zero] : buffer -> view<[%spans]x{CHUNKS}x2xi32>
      %neg_inf = scalar.constant -3.0e38 : f32
      %big = scalar.constant 2147483647 : i32
      %bv, %bi = scf.for %c = [%lane to %chunks step %c256](%v = %neg_inf : f32, %i = %big : i32) -> (f32, i32) {{
        %bounded_c = index.assume %c [range(%c, 0, {CHUNKS - 1})] : index
        %bits_value = view.load %partial_view[%span, %bounded_c, %c0] : view<[%spans]x{CHUNKS}x2xi32> -> i32
        %id = view.load %partial_view[%span, %bounded_c, %c1] : view<[%spans]x{CHUNKS}x2xi32> -> i32
        %value = scalar.bitcast %bits_value : i32 to f32
        %better = scalar.cmpf ogt, %value, %v : f32
        %nv = scf.select %better, %value, %v : f32
        %ni = scf.select %better, %id, %i : i32
        scf.yield %nv, %ni : f32, i32
      }}
      %best = kernel.workgroup.reduce<maxnumf> %bv : f32
      %is_best = scalar.cmpf oeq, %bv, %best : f32
      %candidate = scf.select %is_best, %bi, %big : i32
      %winner = kernel.workgroup.reduce<minsi> %candidate : i32
      %first = index.cmp eq, %lane, %c0 : index
      scf.if %first {{
        // Other end-of-turn ids also end the turn: report the service terminal.
        %eos = view.load %header[1] : view<4xi32> -> i32
{eos_text()}
        %result = buffer.view %outputs[%zero] : buffer -> view<[%spans]xi32>
        view.store %token, %result[%span] : i32, view<[%spans]xi32>
      }}
    }}
  }}
  kernel.return
}}
"""


def linear(fmt, x, s, w, wsize, y, k, n, kval, mode="store"):
    """Projection y = x W^T: WMMA GEMM for 64+ tokens (128-row tiles from 128),
    else the decode matrix-vector kernel (1-token or 4-token variant)."""
    base = f"g_mv_{fmt}_k{kval}" + ("_swiglu" if mode == "swiglu" else "")
    if mode == "swiglu":
        wa, wb = w
        args = f"%t, {k}, {n}, {x}, {s}, {wa}, {wb}, {y}"
        types = f"index, index, index, buffer, buffer, view<{wsize}xi8>, view<{wsize}xi8>, buffer"
        mv_args = f"({n}, %t, %zero_i32, {x}, {wa}, {wb}, {y}) : [index, index](index, index, i32, buffer, view<{wsize}xi8>, view<{wsize}xi8>, buffer)"
    else:
        args = f"%t, {k}, {n}, {x}, {s}, {w}, {y}"
        types = f"index, index, index, buffer, buffer, view<{wsize}xi8>, buffer"
        mv_args = f"({n}, %t, %zero_i32, {x}, {w}, {y}) : [index, index](index, index, i32, buffer, view<{wsize}xi8>, buffer)"
    x16 = {"%h": "%h16", "%o": "%o16", "%f": "%f16"}[x]
    args = args.replace(f", {x}, {s},", f", {x16}, {s},")
    return f"""scf.if %gemm {{
      scf.if %big {{
        kernel.launch @g_gemm_{fmt}_{mode}_m128[%t, {n}]({args}) : [index, index]({types})
      }} else {{
        kernel.launch @g_gemm_{fmt}_{mode}[%t, {n}]({args}) : [index, index]({types})
      }}
    }} else {{
      scf.if %single {{
        kernel.launch @{base}[{n}, %t]{mv_args}
      }} else {{
        kernel.launch @{base}_t4[{n}, %t]{mv_args}
      }}
    }}"""

def head_mv(TG, fmt):
    """Output head argmax: workgroup = 128 vocabulary rows x TG spans,
    one lane per 32-value group of the 6656-wide row with each span's
    activations in registers; weights stream once per group of TG spans."""
    R = 128
    groups = HG
    lanes = (groups + 31) // 32 * 32
    waves = lanes // 32
    rb = row_bytes(fmt, H)
    name = "g_head_mv" + (f"_t{TG}" if TG > 1 else "")
    L = [f"""
kernel.def export("{name}") @{name}(%spans: index) {{
  %c1 = index.constant 1 : index
  %lanes = index.constant {lanes} : index
  %chunks = index.constant {CHUNKS} : index
  %tg = index.constant {TG} : index
  %tg_less = index.constant {TG - 1} : index
  %padded = index.add %spans, %tg_less : index
  %span_groups = index.div %padded, %tg : index
  kernel.launch.config workgroups(%span_groups, %chunks, %c1) workgroup_size(%lanes, %c1, %c1) : index
}} launch(%spans_arg: index, %tokens_arg: index, %metadata: buffer, %h: buffer, %w: buffer, %partials: buffer) {{
  %spans = index.assume %spans_arg [range(%spans_arg, 1, 8192)] : index
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %chunk_id = kernel.workgroup.id<y> : index
  %chunk = index.assume %chunk_id [range(%chunk_id, 0, {CHUNKS - 1})] : index
  %sgroup = kernel.workgroup.id<x> : index
  %workitem_id = kernel.workitem.id<x> : index
  %workitem = index.assume %workitem_id [range(%workitem_id, 0, {lanes - 1})] : index
  %wave_id = kernel.subgroup.id : index
  %wave = index.assume %wave_id [range(%wave_id, 0, {waves - 1})] : index
  %lane_id = kernel.subgroup.lane.id : index
  %lane = index.assume %lane_id [range(%lane_id, 0, 31)] : index
  %zero = index.constant 0 : offset
  %c16o = index.constant 16 : offset
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c8 = index.constant 8 : index
  %tgc = index.constant {TG} : index
  %cW = index.constant {waves} : index
  %rows = index.constant {R} : index
  %groups = index.constant {groups} : index
  %vocab = index.constant {V} : index
  %row_bytes = index.constant {rb} : offset
  %block_bytes = index.constant {FMT[fmt]} : offset
  %zero_f = scalar.constant 0.0 : f32
  %zero32 = vector.constant 0.0 : vector<32xf32>
  %c0_i32 = scalar.constant 0 : i32
  %c1_i32 = scalar.constant 1 : i32
  %false = scalar.constant false : i1
  %header = buffer.view %metadata[%zero] : buffer -> view<4xi32>
  %count32 = view.load %header[0] : view<4xi32> -> i32
  %count_bound = scalar.assume %count32 [range(%count32, 0, 8192)] : i32
  %count = index.cast %count_bound : i32 to index
  %packets = buffer.view %metadata[%c16o] : buffer -> view<[%spans]x9xi32>
  %x_view = buffer.view %h[%zero] : buffer -> view<[%tokens]x{groups}x32xf32>
  %span0 = index.mul %sgroup, %tgc : index
  %ok = index.cmp ult, %workitem, %groups : index
  %g_last = index.constant {groups - 1} : index
  %g_c = index.min %workitem, %g_last : index
  %g = index.assume %g_c [range(%g_c, 0, {groups - 1})] : index
  %cgpb = index.constant {SB[fmt] // 32} : index
  %blk = index.div %g, %cgpb : index
  %within = index.rem %g, %cgpb : index
  %ba = index.scale %blk, %block_bytes : index, offset -> offset
  %first_row = index.mul %chunk, %rows : index"""]
    for j in range(TG):
        L.append(f"""  %sj{j} = index.constant {j} : index
  %span{j}_raw = index.add %span0, %sj{j} : index
  %in_s{j} = index.cmp ult, %span{j}_raw, %spans : index
  %in_c{j} = index.cmp ult, %span{j}_raw, %count : index
  %in{j} = scalar.andi %in_s{j}, %in_c{j} : i1
  %live{j}, %token{j} = scf.if %in{j} -> (i1, index) {{
    %sp = index.assume %span{j}_raw [lt(%span{j}_raw, %spans)] : index
    %flags = view.load %packets[%sp, 6] : view<[%spans]x9xi32> -> i32
    %bits = scalar.andi %flags, %c1_i32 : i32
    %selected = scalar.cmpi ne, %bits, %c0_i32 : i32
    %length32 = view.load %packets[%sp, 0] : view<[%spans]x9xi32> -> i32
    %input32 = view.load %packets[%sp, 4] : view<[%spans]x9xi32> -> i32
    %last32 = scalar.addi %input32, %length32 : i32
    %tok32 = scalar.subi %last32, %c1_i32 : i32
    %tok_low = scalar.maxsi %tok32, %c0_i32 : i32
    %tok_bound = scalar.assume %tok_low [range(%tok_low, 0, 8191)] : i32
    %tok_value = index.cast %tok_bound : i32 to index
    %one_l = index.constant 1 : index
    %tokens_last = index.sub %tokens, %one_l : index
    %tok_safe = index.min %tok_value, %tokens_last : index
    scf.yield %selected, %tok_safe : i1, index
  }} else {{
    scf.yield %false, %c0 : i1, index
  }}
  %use{j} = scalar.andi %live{j}, %ok : i1
  %x{j} = scf.if %use{j} -> (vector<32xf32>) {{
    %tk = index.assume %token{j} [lt(%token{j}, %tokens)] : index
    %v = vector.load %x_view[%tk, %g, %c0] : view<[%tokens]x{groups}x32xf32> -> vector<32xf32>
    scf.yield %v : vector<32xf32>
  }} else {{
    scf.yield %zero32 : vector<32xf32>
  }}""")
    L.append(f"""  %part_bytes = index.constant {waves * R * TG * 4} : offset
  %parts = buffer.alloca<workgroup> align(16) %part_bytes : buffer
  %pv = buffer.view %parts[%zero] : buffer -> view<{waves}x{R}x{TG}xf32>
  %lane_first = index.cmp eq, %lane, %c0 : index
  scf.for %r = [%c0 to %rows step %c1] {{
    %row_value = index.add %first_row, %r : index
    %in_vocab = index.cmp ult, %row_value, %vocab : index
    %load = scalar.andi %in_vocab, %ok : i1
    %last_row = index.constant {V - 1} : index
    %row_c = index.min %row_value, %last_row : index
    %ro = index.scale %row_c, %row_bytes : index, offset -> offset
    %o = index.add %ro, %ba : offset
    %rb = index.assume %r [range(%r, 0, {R - 1})] : index
    %wv = scf.if %load -> (vector<32xf32>) {{
      %v = func.call @g_{fmt}_values(%w, %o, %within) : (buffer, offset, index) -> (vector<32xf32>)
      scf.yield %v : vector<32xf32>
    }} else {{
      scf.yield %zero32 : vector<32xf32>
    }}""")
    for j in range(TG):
        L.append(f"""    %pa{j} = vector.dotf<reassoc|contract> %wv, %x{j}, %zero_f : vector<32xf32>, vector<32xf32>, f32
    %ps{j} = kernel.subgroup.reduce<addf> %pa{j} : f32
    scf.if %lane_first {{
      view.store %ps{j}, %pv[%wave, %rb, {j}] : f32, view<{waves}x{R}x{TG}xf32>
    }}""")
    L.append("  }")
    L.append("  kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    L.append(f"  %partial_view = buffer.view %partials[%zero] : buffer -> view<[%spans]x{CHUNKS}x2xi32>")
    for j in range(TG):
        L.append(f"""  %writer{j} = index.cmp eq, %workitem, %sj{j} : index
  %store{j} = scalar.andi %writer{j}, %live{j} : i1
  scf.if %store{j} {{
    %neg_inf = scalar.constant -3.0e38 : f32
    %best, %best_id = scf.for %r = [%c0 to %rows step %c1](%bv = %neg_inf : f32, %bi = %c0_i32 : i32) -> (f32, i32) {{
      %rb = index.assume %r [range(%r, 0, {R - 1})] : index
      %total = scf.for %ww = [%c0 to %cW step %c1](%acc = %zero_f : f32) -> (f32) {{
        %bw = index.assume %ww [range(%ww, 0, {waves - 1})] : index
        %p = view.load %pv[%bw, %rb, {j}] : view<{waves}x{R}x{TG}xf32> -> f32
        %nx = scalar.addf %acc, %p : f32
        scf.yield %nx : f32
      }}
      %row_value = index.add %first_row, %r : index
      %in_vocab = index.cmp ult, %row_value, %vocab : index
      %greater = scalar.cmpf ogt, %total, %bv : f32
      %better = scalar.andi %greater, %in_vocab : i1
      %row32 = index.cast %row_value : index to i32
      %v = scf.select %better, %total, %bv : f32
      %i = scf.select %better, %row32, %bi : i32
      scf.yield %v, %i : f32, i32
    }}
    %ss = index.assume %span{j}_raw [lt(%span{j}_raw, %spans)] : index
    %bits_value = scalar.bitcast %best : f32 to i32
    view.store %bits_value, %partial_view[%ss, %chunk, %c0] : i32, view<[%spans]x{CHUNKS}x2xi32>
    view.store %best_id, %partial_view[%ss, %chunk, %c1] : i32, view<[%spans]x{CHUNKS}x2xi32>
  }}""")
    L.append("  kernel.return\n}\n")
    return "\n".join(L)


USED_MV = set()      # (fmt, K, swiglu) decode matrix-vector kernels the model launches
USED_GEMM = {}       # fmt -> set of GEMM modes


def lin(fmt, x, s, w, wsize, y, k, n, kval, mode="store"):
    USED_MV.add((fmt, kval, mode == "swiglu"))
    USED_GEMM.setdefault(fmt, set()).add(mode)
    return linear(fmt, x, s, w, wsize, y, k, n, kval, mode)


def model():
    """Command program: embed, then a statically unrolled layer loop (each
    layer's tensor formats come from the GGUF), then the argmax head."""
    QD, KD = QH * D, KVH * D
    sliding = set(CFG.get("sliding_layers", []))
    nope = set(CFG.get("nope_layers", []))
    head_name = "output.weight" if "output.weight" in TENS else "token_embd.weight"
    head_fmt = fmt_of(head_name)
    hsize = tensor_bytes(head_name)
    esize = tensor_bytes("token_embd.weight")
    L_ = [f"""
config.decl @llm.context : %value: index where [range(%value, 16, 262144)]

config.decl @llm.spans : %value: index where [range(%value, 1, 8192)]

config.decl @llm.tokens : %value: index where [range(%value, 1, 8192)]

config.decl @llm.pool : %value: index where [range(%value, 16, 1048576)]

command.program.def public @llm_advance() launch(%weights: buffer, %residual: buffer, %metadata: buffer, %origins: buffer, %arena: buffer, %tokens: buffer, %outputs: buffer) {{
  %t = config.get @llm.tokens : index
  %spans = config.get @llm.spans : index
  %context = config.get @llm.context : index
  %pool = config.get @llm.pool : index
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %nH = index.constant {H} : index
  %nFF = index.constant {FF} : index
  %nQ = index.constant {QD} : index
  %nKV = index.constant {KD} : index
  %zero_i32 = scalar.constant 0 : i32
  %one_i32 = scalar.constant 1 : i32
  %row_bytes = index.constant {H * 4} : offset
  %wide_bytes = index.constant {QD * 4} : offset
  %kv_bytes = index.constant {KD * 4} : offset
  %info_bytes = index.constant 16 : offset
  %ffn_bytes = index.constant {FF * 4} : offset
  %partial_bytes = index.constant {CHUNKS * 8} : offset
  %x_size = index.scale %t, %row_bytes : index, offset -> offset
  %q_size = index.scale %t, %wide_bytes : index, offset -> offset
  %kv_size = index.scale %t, %kv_bytes : index, offset -> offset
  %info_size = index.scale %t, %info_bytes : index, offset -> offset
  %ffn_size = index.scale %t, %ffn_bytes : index, offset -> offset
  %partial_size = index.scale %spans, %partial_bytes : index, offset -> offset
  %x = buffer.alloca<global> align(256) %x_size : buffer
  %h = buffer.alloca<global> align(256) %x_size : buffer
  %y = buffer.alloca<global> align(256) %x_size : buffer
  %q = buffer.alloca<global> align(256) %q_size : buffer
  %o = buffer.alloca<global> align(256) %q_size : buffer
  %k = buffer.alloca<global> align(256) %kv_size : buffer
  %v = buffer.alloca<global> align(256) %kv_size : buffer
  %info = buffer.alloca<global> align(256) %info_size : buffer
  %f = buffer.alloca<global> align(256) %ffn_size : buffer
  %partials = buffer.alloca<global> align(256) %partial_size : buffer
  // 64 splits per head for decode, 16 for the prefill fallback.
  %c64 = index.constant 64 : index
  %c128 = index.constant 128 : index
  %prefill_shape = index.cmp uge, %t, %c64 : index
  %pacc_dec = index.constant {QH * 64 * 128 * 4} : offset
  %pacc_pre = index.constant {QH * 16 * 128 * 4} : offset
  %pml_dec = index.constant {QH * 64 * 2 * 4} : offset
  %pml_pre = index.constant {QH * 16 * 2 * 4} : offset
  %pacc_bytes = scf.select %prefill_shape, %pacc_pre, %pacc_dec : offset
  %pml_bytes = scf.select %prefill_shape, %pml_pre, %pml_dec : offset
  %pacc_size = index.scale %t, %pacc_bytes : index, offset -> offset
  %pml_size = index.scale %t, %pml_bytes : index, offset -> offset
  %pacc = buffer.alloca<global> align(256) %pacc_size : buffer
  %pml = buffer.alloca<global> align(256) %pml_size : buffer
  %gemm = index.cmp uge, %t, %c64 : index
  %single = index.cmp eq, %t, %c1 : index
  %big = index.cmp uge, %t, %c128 : index
  %scale_entry = index.constant 8 : offset
  %scale_size = index.scale %t, %scale_entry : index, offset -> offset
  %s_h = buffer.alloca<global> align(256) %scale_size : buffer
  %s_o = buffer.alloca<global> align(256) %scale_size : buffer
  %s_f = buffer.alloca<global> align(256) %scale_size : buffer
  %half_row = index.constant {H * 2} : offset
  %half_wide = index.constant {QD * 2} : offset
  %half_ffn = index.constant {FF * 2} : offset
  %h16_size = index.scale %t, %half_row : index, offset -> offset
  %o16_size = index.scale %t, %half_wide : index, offset -> offset
  %f16_size = index.scale %t, %half_ffn : index, offset -> offset
  %h16 = buffer.alloca<global> align(256) %h16_size : buffer
  %o16 = buffer.alloca<global> align(256) %o16_size : buffer
  %f16 = buffer.alloca<global> align(256) %f16_size : buffer
"""]
    if GATE:
        L_.append("  %gate = buffer.alloca<global> align(256) %q_size : buffer\n")
    if not QK_NORM:
        L_.append("  %no_norm = buffer.alloca<global> align(256) %row_bytes : buffer\n")
    L_.append(f"""
  kernel.launch @g_token_info[%t](%t, %spans, %metadata, %info) : [index](index, index, buffer, buffer)
  %list_entry = index.constant 4 : offset
  %list_size = index.scale %t, %list_entry : index, offset -> offset
  %count_size = index.constant 256 : offset
  %fallback_list = buffer.alloca<global> align(256) %list_size : buffer
  %fallback_count = buffer.alloca<global> align(256) %count_size : buffer
  scf.if %gemm {{
    kernel.launch @g_attention_fallback_list[%t](%t, %info, %fallback_list, %fallback_count) : [index](index, buffer, buffer, buffer)
  }}
  %embedding = command.parameter %weights, "token_embd.weight" : view<{esize}xi8>
  %first_norm = command.parameter %weights, "blk.0.attn_norm.weight" : view<{H}xf32>
  kernel.launch @g_embed[%t](%t, %tokens, %embedding, %first_norm, %x, %h) : [index](index, buffer, view<{esize}xi8>, view<{H}xf32>, buffer, buffer)
  %output_norm = command.parameter %weights, "output_norm.weight" : view<{H}xf32>
""")
    gate = "%gate" if GATE else "%q"
    for i in range(L):
        b = f"blk.{i}."
        nm = lambda n: b + n + ".weight"
        sz = lambda n: tensor_bytes(nm(n))
        par = lambda var, n, ty=None: f'    %{var}_{i} = command.parameter %weights, "{nm(n)}" : {ty or f"view<{sz(n)}xi8>"}\n'
        T = [f"  // ---- layer {i}{' (sliding)' if i in sliding else ''}{' (no RoPE)' if i in nope else ''}\n  scf.if %gemm {{\n    kernel.launch @g_row_scale_f16[%t](%t, %nH, %h, %s_h, %h16) : [index](index, index, buffer, buffer, buffer)\n  }}\n",
             f"  %layer_{i} = index.constant {i} : index\n  %sliding_{i} = scalar.constant {1 if i in sliding else 0} : i32\n  %rope_{i} = scalar.constant {0 if i in nope else 1} : i32\n"]
        T += [par("wq", "attn_q"), par("wk", "attn_k"), par("wv", "attn_v"), par("wo", "attn_output"),
              par("w_gate", "ffn_gate"), par("w_up", "ffn_up"), par("w_down", "ffn_down"),
              par("ffn_norm", "ffn_norm", f"view<{H}xf32>")]
        if GATE:
            T.append(par("wg", "attn_gate"))
        if QK_NORM:
            T += [par("q_norm", "attn_q_norm", "view<128xf32>"), par("k_norm", "attn_k_norm", "view<128xf32>")]
        if POST_NORM:
            T += [par("post_attn", "post_attention_norm", f"view<{H}xf32>"), par("post_ffn", "post_ffw_norm", f"view<{H}xf32>")]
        T.append("  " + lin(fmt_of(nm("attn_q")), "%h", "%s_h", f"%wq_{i}", sz("attn_q"), "%q", "%nH", "%nQ", H) + "\n")
        T.append("  " + lin(fmt_of(nm("attn_k")), "%h", "%s_h", f"%wk_{i}", sz("attn_k"), "%k", "%nH", "%nKV", H) + "\n")
        T.append("  " + lin(fmt_of(nm("attn_v")), "%h", "%s_h", f"%wv_{i}", sz("attn_v"), "%v", "%nH", "%nKV", H) + "\n")
        if GATE:
            T.append("  " + lin(fmt_of(nm("attn_gate")), "%h", "%s_h", f"%wg_{i}", sz("attn_gate"), "%gate", "%nH", "%nQ", H) + "\n")
        qn, kn = (f"%q_norm_{i}", f"%k_norm_{i}") if QK_NORM else ("%no_norm", "%no_norm")
        nty = "view<128xf32>" if QK_NORM else "buffer"
        T.append(f"""  kernel.launch @g_qk_rope_store[%t](%t, %layer_{i}, %rope_{i}, %context, %pool, %q, %k, %v, {qn}, {kn}, %info, %origins, %arena) : [index](index, index, i32, index, index, buffer, buffer, buffer, {nty}, {nty}, buffer, buffer, buffer)
  scf.if %gemm {{
    kernel.launch @g_attention_flash2[%t](%t, %layer_{i}, %sliding_{i}, %context, %pool, %q, {gate}, %info, %origins, %arena, %o) : [index](index, index, i32, index, index, buffer, buffer, buffer, buffer, buffer, buffer)
    kernel.launch @g16_attention_split_listed[%fallback_count](%t, %fallback_list, %layer_{i}, %sliding_{i}, %context, %pool, %q, %info, %origins, %arena, %pacc, %pml) : [buffer](index, buffer, index, i32, index, index, buffer, buffer, buffer, buffer, buffer, buffer)
    kernel.launch @g16_attention_combine_listed[%fallback_count](%t, %fallback_list, %sliding_{i}, %info, %pacc, %pml, {gate}, %o) : [buffer](index, buffer, i32, buffer, buffer, buffer, buffer, buffer)
  }} else {{
    kernel.launch @g_attention_split[%t](%t, %layer_{i}, %sliding_{i}, %context, %pool, %q, %info, %origins, %arena, %pacc, %pml) : [index](index, index, i32, index, index, buffer, buffer, buffer, buffer, buffer, buffer)
    kernel.launch @g_attention_combine[%t](%t, %sliding_{i}, %info, %pacc, %pml, {gate}, %o) : [index](index, i32, buffer, buffer, buffer, buffer, buffer)
  }}
  scf.if %gemm {{
    kernel.launch @g_row_scale_f16[%t](%t, %nQ, %o, %s_o, %o16) : [index](index, index, buffer, buffer, buffer)
  }}
""")
        T.append("  " + lin(fmt_of(nm("attn_output")), "%o", "%s_o", f"%wo_{i}", sz("attn_output"), "%y", "%nQ", "%nH", QD) + "\n")
        pa = f"%post_attn_{i}" if POST_NORM else f"%ffn_norm_{i}"
        T.append(f"  kernel.launch @g_post_add_norm[%t](%t, %y, {pa}, %x, %ffn_norm_{i}, %h) : [index](index, buffer, view<{H}xf32>, buffer, view<{H}xf32>, buffer)\n")
        T.append(f"  scf.if %gemm {{\n    kernel.launch @g_row_scale_f16[%t](%t, %nH, %h, %s_h, %h16) : [index](index, index, buffer, buffer, buffer)\n  }}\n")
        fg, fu = fmt_of(nm("ffn_gate")), fmt_of(nm("ffn_up"))
        assert fg == fu, f"layer {i}: ffn_gate {fg} != ffn_up {fu} (split the swiglu linear)"
        T.append("  " + lin(fg, "%h", "%s_h", (f"%w_gate_{i}", f"%w_up_{i}"), sz("ffn_gate"), "%f", "%nH", "%nFF", H, mode="swiglu") + "\n")
        T.append(f"  scf.if %gemm {{\n    kernel.launch @g_row_scale_f16[%t](%t, %nFF, %f, %s_f, %f16) : [index](index, index, buffer, buffer, buffer)\n  }}\n")
        T.append("  " + lin(fmt_of(nm("ffn_down")), "%f", "%s_f", f"%w_down_{i}", sz("ffn_down"), "%y", "%nFF", "%nH", FF) + "\n")
        pf = f"%post_ffn_{i}" if POST_NORM else f"%ffn_norm_{i}"
        if i == L - 1:
            nn = "%output_norm"
        else:
            T.append(f'  %next_norm_{i} = command.parameter %weights, "blk.{i + 1}.attn_norm.weight" : view<{H}xf32>\n')
            nn = f"%next_norm_{i}"
        T.append(f"  kernel.launch @g_post_add_norm[%t](%t, %y, {pf}, %x, {nn}, %h) : [index](index, buffer, view<{H}xf32>, buffer, view<{H}xf32>, buffer)\n")
        L_.append("".join(T).replace("\n    %", "\n  %").replace("\n      ", "\n    "))
    L_.append(f"""
  %head = command.parameter %weights, "{head_name}" : view<{hsize}xi8>
  %one_span = index.cmp eq, %spans, %c1 : index
  scf.if %one_span {{
    kernel.launch @g_head_mv[%spans](%spans, %t, %metadata, %h, %head, %partials) : [index](index, index, buffer, buffer, view<{hsize}xi8>, buffer)
  }} else {{
    kernel.launch @g_head_mv_t4[%spans](%spans, %t, %metadata, %h, %head, %partials) : [index](index, index, buffer, buffer, view<{hsize}xi8>, buffer)
  }}
  kernel.launch @g_head_select[%spans](%spans, %metadata, %partials, %outputs) : [index](index, buffer, buffer, buffer)
  command.return
}}
""")
    return "".join(L_), head_fmt


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.environ.get("LLM_OUT") or os.path.join(here, "out")
    os.makedirs(out_dir, exist_ok=True)
    program, head_fmt = model()  # first: records the mv / GEMM variants it launches
    parts = [f"// Generated by gen_llm.py from {os.environ.get('LLM_CONFIG') or sys.argv[1]}.\n", COMMON.replace("%c8_q6", "%c8q6").replace(
        "  %c64 = index.constant 64 : index\n  %qh_base", "  %c64 = index.constant 64 : index\n  %c8q6 = index.constant 8 : index\n  %qh_base")]
    for fmt in ("q4_k", "q5_k", "q6_k", "q8_0"):
        parts.append(row_dot(fmt))
    for TG in (1, 4):
        for fmt, K, sw in sorted(USED_MV):
            parts.append(mv(fmt, K, TG=TG, swiglu=sw))
    parts += [NORMS, ROW_SCALE, qk_rope(), attention(), head(), head_mv(1, head_fmt), head_mv(4, head_fmt)]
    env = dict(os.environ, GEMM_FORMATS=",".join(f"{f}:{'+'.join(sorted(m))}" for f, m in sorted(USED_GEMM.items())))
    # Prefill GEMM switches, ON by default; set GEMM_N_FASTEST=0 / GEMM_SKIP_PADDING=0 to A/B them off.
    env.setdefault("GEMM_SKIP_PADDING", "1"); env.setdefault("GEMM_N_FASTEST", "1")
    parts.append(subprocess.run([sys.executable, os.path.join(here, "gen_gemm.py")], capture_output=True, text=True, check=True, env=env).stdout)
    parts.append(open(os.path.join(here, "tile_uniform.loom")).read())
    flash2 = os.path.join(out_dir, "flash2.loom")
    subprocess.run([sys.executable, os.path.join(here, "gen_flash2.py")], check=True,
                   env=dict(os.environ, FLASH2_TARGET="llm", FLASH2_OUT=flash2, LLM_CONFIG=os.environ.get("LLM_CONFIG") or sys.argv[1]))
    parts.append(open(flash2).read())
    parts.append(program)
    out = os.path.join(out_dir, "model.loom")
    open(out, "w").write("\n".join(parts))
    print("wrote", out)
