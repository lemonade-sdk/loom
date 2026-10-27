# Generates decode.loom: split-K decode attention with U keys per iteration.
import os, re
U = int(os.environ.get("DEC_UNROLL", "16"))
S = int(os.environ.get("DEC_SPLITS", "16"))
src = open(os.path.join(os.path.dirname(__file__), "decode_base.loom.txt")).read()

body = []
for i in range(U):
    body.append(f"""      %key{i}_raw = index.add %j, %c{i}u : index
      %ok{i} = index.cmp ult, %key{i}_raw, %stop : index
      %key{i}_c = scf.select %ok{i}, %key{i}_raw, %j : index
      %kp{i} = index.add %window_low, %key{i}_c : index
      %ko{i} = func.call @flint_kv_offset(%origins, %row, %kp{i}, %blocks, %k_plane, %pool) : (buffer, index, index, index, index, index) -> (offset)
      %kc{i} = buffer.view %arena[%ko{i}] : buffer -> view<4x32x4xf16>
      %kh{i} = vector.load %kc{i}[%kv_head, %lane, %c0] : view<4x32x4xf16> -> vector<4xf16>
      %k{i} = vector.extf %kh{i} : vector<4xf16> to vector<4xf32>
      %vo{i} = index.add %ko{i}, %plane_stride : offset
      %vc{i} = buffer.view %arena[%vo{i}] : buffer -> view<4x32x4xf16>
      %vh{i} = vector.load %vc{i}[%kv_head, %lane, %c0] : view<4x32x4xf16> -> vector<4xf16>
      %v{i} = vector.extf %vh{i} : vector<4xf16> to vector<4xf32>
""")
for i in range(U):
    body.append(f"""      %prod{i} = vector.mulf %q4, %k{i} : vector<4xf32>
      %part{i} = vector.reduce<addf> %prod{i}, %zero_f : vector<4xf32>, f32
      %sr{i} = kernel.subgroup.reduce<addf> %part{i} : f32
      %s{i} = scf.select %ok{i}, %sr{i}, %neg_big : f32
""")
prev = "%m"
for i in range(U):
    body.append(f"      %mx{i} = scalar.maxnumf {prev}, %s{i} : f32\n"); prev = f"%mx{i}"
body.append(f"""      %m_shift = scalar.subf %m, {prev} : f32
      %alpha = scalar.expf<afn> %m_shift : f32
      %l_scaled = scalar.mulf %l, %alpha : f32
      %alpha_v = vector.splat %alpha : vector<4xf32>
      %acc_s = vector.mulf %acc, %alpha_v : vector<4xf32>
""")
pl, pa = "%l_scaled", "%acc_s"
for i in range(U):
    body.append(f"""      %d{i} = scalar.subf %s{i}, {prev} : f32
      %p{i} = scalar.expf<afn> %d{i} : f32
      %la{i} = scalar.addf {pl}, %p{i} : f32
      %pv{i} = vector.splat %p{i} : vector<4xf32>
      %aa{i} = vector.fmaf %pv{i}, %v{i}, {pa} : vector<4xf32>
"""); pl, pa = f"%la{i}", f"%aa{i}"
body.append(f"      scf.yield {prev}, {pl}, {pa} : f32, f32, vector<4xf32>\n")
consts = "".join(f"  %c{i}u = index.constant {i} : index\n" for i in range(U))
out = src.replace("@@BODY@@", "".join(body)).replace("@@CONSTS@@", consts).replace("@@U@@", str(U))
def render(S, P):
    return template.replace("@@S@@", str(S)).replace("@@S1@@", str(S - 1)).replace("@@P@@", P)
template = out
out = render(int(os.environ.get("DEC_SMAX", "64")), "flint")
listed_src = render(16, "flint16")
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


split_l = listed(listed_src, "flint16_attention_split", f"""  %c1 = index.constant 1 : index
  %c4 = index.constant 4 : index
  %cS = index.constant 16 : index
  %c256 = index.constant 256 : index
  kernel.launch.config workgroups(%count, %c4, %cS) workgroup_size(%c256, %c1, %c1) : index""")
combine_l = listed(listed_src, "flint16_attention_combine", """  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  kernel.launch.config workgroups(%count, %c32, %c1) workgroup_size(%c32, %c1, %c1) : index""")
count16 = listed_src[listed_src.index("// Splits for"):listed_src.index('kernel.def export("flint16_attention_split")')]
out = out + "\n" + count16 + split_l + "\n" + combine_l
open(os.path.join(os.path.dirname(__file__), "decode.loom"), "w").write(out)
