"""Kernel A/B helper: put the OLD build's version of a kernel next to the NEW
build's under a different name, so one check.case can launch both on identical
inputs.

usage: python ab_kernel.py OLD/program/llm.loom NEW/program/llm.loom KERNEL OUT.loom-test
then append your check.case(s) to OUT.loom-test and run `loom-test OUT.loom-test --device=amdgpu`.

The old kernel is renamed KERNEL_old (symbol and export). Helper funcs it calls
are taken from the NEW file, so if your change edited a shared helper, A/B the
helper's callers from two complete files instead.
Every A/B needs a NEGATIVE CONTROL, otherwise two kernels that both write nothing
"agree": a second case asserting the new output equals its zero fill, which must
FAIL. Run: loom-test OUT.loom-test --device=amdgpu --xfail=@ab_negative_control=EXPECT/003
(the run exits 0 only if the A/B case passes and the control fails).

Case skeleton (fill shapes from the kernel's launch signature). Kernel params typed
`buffer` are passed as tensors, and the launch signature spells them as tensor types
(tensor<192x2560xf32>, tensor<NBYTESxi8>, ...). GEMM kernels also take the row-scale
inputs written by g_row_scale_f16: f16 activations [rows]xK and `scales` [rows]x2xf32.
Scales must be NON-ZERO for live rows, because padding-free GEMMs skip rows with
scale 0. The easiest source is to launch g_row_scale_f16 on your f32 x inside the case.
  check.case public @ab_KERNEL {
    %n = check.literal value(4096) : index
    %t = check.literal value(192) : index
    %x = check.generate.iota offset(-1.0) step(0.0003) : tensor<192x2560xf32>
    %w = check.file.read.npy path("w.npy") : tensor<...xi8>          // real GGUF bytes
    %y_new = check.generate.fill value(0.0) : tensor<192x4096xf32>
    %y_old = check.generate.fill value(0.0) : tensor<192x4096xf32>
    kernel.launch @KERNEL[...](... %y_new) : [...](...)
    kernel.launch @KERNEL_old[...](... %y_old) : [...](...)
    check.expect.close actual(%y_new) expected(%y_old) atol(1.0e-3) rtol(1.0e-3) nan(different) : tensor<192x4096xf32>
    check.return
  }
  check.case public @ab_negative_control {
    ... same inputs, launch only @KERNEL into %y_new ...
    %zeros = check.generate.fill value(0.0) : tensor<192x4096xf32>
    check.expect.close actual(%y_new) expected(%zeros) atol(0.0) rtol(0.0) nan(different) : tensor<192x4096xf32>
    check.return
  }
"""
import sys

old_path, new_path, name, out = sys.argv[1:5]


def extract(text, kname):
    a = text.index(f'kernel.def export("{kname}") @{kname}(')
    end = text.index("kernel.return\n}\n", a) + len("kernel.return\n}\n")
    return text[a:end]


old = extract(open(old_path).read(), name)
old = old.replace(f'export("{name}") @{name}(', f'export("{name}_old") @{name}_old(', 1)
new_text = open(new_path).read()
extract(new_text, name)  # must exist in the new build too
open(out, "w", newline="\n").write(new_text + f"\n// ---- A/B: {name} from {old_path}\n" + old + "\n")
print(f"wrote {out}: new library + @{name}_old; append check.case records and run loom-test")
