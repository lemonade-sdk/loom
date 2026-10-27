"""Builds the Q4_1 (int4) Flint package from the Q4_K_M sources."""
import os, re, subprocess, json, shutil
here = os.path.dirname(os.path.abspath(__file__))
src, pkg = os.path.join(here, "program"), os.path.join(here, "package_int4")
os.makedirs(os.path.join(pkg, "program"), exist_ok=True)
os.makedirs(os.path.join(pkg, "parameter"), exist_ok=True)
read = lambda f: open(os.path.join(src, f)).read()

def to_q4_1(text):
    text = text.replace("@flint_q8_0_row_dot(", "@flint_q4_1_row_dot(").replace("@flint_q4_k_row_dot(", "@flint_q4_1_row_dot(")
    for old, new in [("2176", "1280"), ("1088", "640"), ("1152", "1280"), ("288", "320"), ("589824", "655360")]:
        text = text.replace(f"index.constant {old} : offset", f"index.constant {new} : offset")
    return text

def model(text):
    text = text.replace('%embedding = command.parameter %weights, "token_embd.weight" : view<326713344xi8>', '%embedding = command.parameter %weights, "token_embd.weight" : view<307494912xf32>')
    text = text.replace("kernel.launch @flint_embed[%t](%t, %tokens, %embedding, %x) : [index](index, buffer, view<326713344xi8>, buffer)", "kernel.launch @flint_embed_f32[%t](%t, %tokens, %embedding, %x) : [index](index, buffer, view<307494912xf32>, buffer)")
    text = re.sub(r"@flint_(matmul|gemm)_q(8_0|4_k)", r"@flint_\1_q4_1", text)
    for old, new in [("8912896", "5242880"), ("1114112", "655360"), ("7077888", "7864320"), ("150994944", "167772160"), ("2228224", "1310720"), ("326713344", "192184320")]:
        text = text.replace(f"view<{old}xi8>", f"view<{new}xi8>")
    return text.replace("%moe_r = index.constant 32 : index", f"%moe_r = index.constant {os.environ.get('MOE_TILE', '32')} : index")

env = dict(os.environ, FLINT_FORMATS="q4_1", FLINT_MOE_FMT="q4_1", GEMM_PREFIX="flint", GEMM_KMETA="@flint_q4_k_scaled_metadata", GEMM_FORMATS="q4_1:store+acc+swiglu", GEMM_SKIP_PADDING="1", GEMM_N_FASTEST=os.environ.get("GEMM_N_FASTEST", "1"))
MOE_TILE = os.environ.get("MOE_TILE", "32")
env["MOE_TILE"] = MOE_TILE
gemm = subprocess.run(["python3", os.path.join(src, "gen_gemm.py")], env=env, capture_output=True, text=True, check=True).stdout
moe = subprocess.run(["python3", os.path.join(src, "gen_moe.py")], env=env, capture_output=True, text=True, check=True).stdout
lib = "\n".join([read("linear.loom"), read("q4_1.loom"), to_q4_1(read("layers.loom")), to_q4_1(read("fused.loom")), gemm, moe, read("flash.loom"), read("flash2.loom"), read("decode.loom"), model(read("model.loom"))])
open(os.path.join(pkg, "program", "flint.loom"), "w").write(lib)
for f in ("prepare.loom", "control.loom", "storage.loom", "service.loom"):
    shutil.copy(os.path.join(src, f), os.path.join(pkg, "program", f))
shutil.copy(os.path.join(src, "weights.loom"), pkg)
shutil.copy(os.environ["FLINT_TOKENIZER"], pkg)  # HF tokenizer.json of RecursiveSI/flint
target = os.path.realpath(os.environ["FLINT_GGUF"])  # the int4 GGUF
link = os.path.join(pkg, "parameter", "flint-int4-q4_1.gguf")
if not os.path.exists(link): os.symlink(target, link)
card = json.load(open(os.path.join(here, "package", "model.json")))
card["model"]["name"] = "Recursive Flint (Loom port, GGUF int4 Q4_1)"
v = card["variants"].pop("q4-k-m")
v["services"]["chat"]["parameters"]["model"]["files"] = [{"path": "parameter/flint-int4-q4_1.gguf", "format": "gguf"}]
card["variants"]["int4"] = v
card["presets"]["chat"]["variant"] = "int4"
card["presets"]["chat"]["execution"]["prefill_capacity"] = int(os.environ.get("PREFILL_CAP", "4096"))
json.dump(card, open(os.path.join(pkg, "model.json"), "w"), indent=1)
print("built", pkg)
