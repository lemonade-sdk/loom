# Loom IR cookbook for model ports

The upstream language guide is in hrx-system at `loom/docs/src/guide/*.md`; read
`kernels-and-launch`, `buffers-views-memory`, `facts-and-specialization`,
`vectors-and-structured-compute`, and `command-programs`. This file collects what
mattered in practice when writing LLM kernels, plus the errors hit most often and
how to fix them.

## Shape of a kernel

```
kernel.def export("g_name") @g_name(%n: index, %tokens: index) {          // grid-shape args
  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  kernel.launch.config workgroups(%n, %tokens, %c1) workgroup_size(%c32, %c1, %c1) : index
} launch(%n_arg: index, %tokens_arg: index, %x: buffer, %w: buffer, %y: buffer) {   // runtime args
  %n = index.assume %n_arg [range(%n_arg, 1, 65536)] : index                       // FACTS: every bound you index with
  %tokens = index.assume %tokens_arg [range(%tokens_arg, 1, 8192)] : index
  %row_id = kernel.workgroup.id<x> : index
  %row = index.assume %row_id [lt(%row_id, %n)] : index
  %lane = kernel.workitem.id<x> : index            // also kernel.subgroup.id / kernel.subgroup.lane.id
  %zero = index.constant 0 : offset
  %x_view = buffer.view %x[%zero] : buffer -> view<[%tokens]x[%groups]x32xf32>
  ...
  kernel.return
}
```

Launching from a command program:
```
kernel.launch @g_name[%n, %t](%n, %t, %x, %w, %y) : [index, index](index, index, buffer, buffer, buffer)
```
- The `[...]` operands feed the grid-shape args. The `(...)` types must match the
  `launch(...)` signature exactly; for weights that is
  `view<BYTESxi8>` from `command.parameter`.
- **Indirect dispatch**: declare the kernel as `@k(%count_buffer: buffer)`, read the
  count with `view.load` in the config region, and launch it as
  `kernel.launch @k[%count_buf](...) : [buffer](...)`. The fallback-attention token
  list uses this.
- **Weights**: `%w = command.parameter %weights, "blk.3.attn_q.weight" : view<Nxi8>`.
  N must equal the tensor's byte size exactly; `gen_llm.tensor_bytes()` computes it.
  Indexed names work as well (`"blk.{}.attn_q.weight"[%layer]`), but every layer then
  needs the same size, which mixed Q4_K_M breaks. Use a static per-layer unroll.
- `config.decl @llm.tokens : %value: index where [range(...)]` declares a stage
  constant that `prepare.loom` sets per packing shape (`config_i64`). Read it with
  `config.get`.

## Proof facts (SUBRANGE/*)

Every view access must be provably in bounds, and the compiler's prover only knows
what you tell it.
- `index.assume %v [range(%v, lo, hi)]`, `[lt(%v, %n)]`, `multiple_of(%v, 16)`. Facts
  attach to the *new* SSA value, so use that value afterwards.
- For scalars use `scalar.assume %i32 [range(...)]` and then `index.cast`.
- `SUBRANGE/010` ("full-vector footprint upper bound is not proven") on
  `vector.load`, `vector.store` or `fragment.store`: the prover needs
  `origin + vector_extent <= view_bound`. There are three fixes:
  - **Clamp.** `%c = index.min %i, %last` followed by
    `index.assume %c [range(%c, 0, last)]`.
  - **Guard.** Wrap the access in `scf.if %in_bounds { ... }`. For prefetch of
    step k+1, use `scf.if step+1 < k_steps`.
  - **Reshape.** Make the view's static dims match the access, e.g.
    `view<[%tokens]x16x32xf32>` loaded with `[t, g, 0] -> vector<32xf32>`.
- `SUBRANGE/023` ("origin lower bound … 0 <= origin"): use `scalar.maxsi %x, %zero`,
  or add a `range(…, 0, …)` fact, before the cast.
- Loop induction variables carry no facts. Re-`assume` them inside the body
  (`%bs = index.assume %s [lt(%s, %spans)]`).
- Values read from buffers (token ids, slots, positions) carry no facts at all.
  Clamp them (`maxsi`/`minsi`) and then `scalar.assume`.

## Other diagnostics

| Code | Meaning | Fix |
|---|---|---|
| PARSE/002 "SSA value '%x' is already defined" | Names are unique per function, **including nested regions**. | Suffix per unrolled copy (`%wq_{i}`). Rename in helpers (`%info_view` → `%token_info`). |
| PARSE/001 "undefined SSA value" | Usually an f-string or generator substitution miss (`{VAR}` left in the text, or a constant defined only in one branch). | `assert "{" not in line` after substitution. Define constants at the top of the kernel. |
| PARSE/003 "unexpected token '{'" | An unescaped `{` in an f-string, or a doubled `}}` in a non-f-string. | Keep `{{`/`}}` only inside f-strings. Generators that mix both should post-substitute placeholders. |
| TYPE/001 "type view<[%tokens]x64x32xf32> does not match contract argument" | `func.call` argument view types must match the callee *spelling*: dynamic `[%groups]` vs static `64`. | Build the view with the same dims the callee declares. |
| TYPE/003 "operand has type offset, expected index" | `offset` (bytes) and `index` (elements) are distinct. | `index.scale %i, %bytes : index, offset -> offset`. Use `index.add` on offsets. |
| TARGET/003 "rejected 'index.div' address-width 'u32'" | 32-bit address math failed because the prover could not bound the division. | Add range facts on the operands. |
| TARGET/092 "rejected 'buffer.alloca'" | `buffer.alloca<workgroup>` with a non-constant size, or inside a func that is not inlined. | Use constant sizes and do the alloca in the kernel body. |
| SYMBOL/002 "referenced but not defined" | A helper func lives in another file that is not concatenated or linked. | Include it in the library list (`model.json` `programs.source.libraries`) or the test's `cat`. |
| BACKEND/005 unspillable-register-exhausted | Too many live vectors (NPU especially). | Smaller unroll, fewer simultaneous accumulators. |

## Idioms that compile to fast code on RDNA3

- **Matvec, one lane per 32-value group.** `vector.load` 32 f32 of x into registers,
  dequantize the 32 weights, then
  `vector.dotf<reassoc|contract> %w, %x, %zero_f`, then
  `kernel.subgroup.reduce<addf>`. Fold Q4_K scale/min per group:
  `d·dot(codes,x) − m·sum(x)`.
- **WMMA.** Use `vector.fragment<lhs|rhs|init>`, `vector.fragment.load<lhs>`, and
  `vector.mma` on f16 operands with f32 accumulation (16×16×16).
  - RDNA3 result layout: element i of lane l is row `2i + l/16`, column `l % 16`.
  - The transposed form (Sᵀ = K·Qᵀ) makes per-query softmax lane-local.
- **Workgroup memory.** Use `buffer.alloca<workgroup> align(16) <const> : buffer` and
  `kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)`.
- **Atomics.** `view.atomic.rmw<addi> %v, %view[...] {ordering = relaxed, scope = workgroup}`.
- **Cross-lane.** `kernel.subgroup.reduce<addf|maxnumf>`, and shuffles with
  `shuffle<xor>` (an xor-16 shuffle swaps the two half-waves).
- **Math.**
  - `scalar.siluf<afn>`, `vector.logisticf<afn>`, `scalar.rsqrtf`, `scalar.expf<afn>`,
    `scalar.cosf<afn>`, `scalar.sinf<afn>`.
  - Range-reduce RoPE angles (subtract a multiple of 2π) before `cosf`/`sinf<afn>`.
  - Use `scalar.fptrunc` for f16 KV writes.
- **Loops.** `scf.for %i = [%a to %b step %s](%acc = %init : f32) -> (f32) { … scf.yield … }`
  handles loop-carried values. Add `unroll` for static trip counts.
  - `scf.if %c -> (T) { scf.yield … } else { … }` must yield in both branches.
  - `scf.select` is a select, not a branch.

## Testing kernels (`check.case`, loom-test)

```
check.case public @mv_q6 {
  %n = check.literal value(1024) : index
  %t = check.literal value(4) : index
  %acc = check.literal value(0) : i32
  %w = check.file.read.npy path("w.npy") : tensor<860160xi8>        // raw GGUF bytes
  %x = check.file.read.npy path("x.npy") : tensor<4x1024xf32>
  %e = check.file.read.npy path("y.npy") : tensor<4x1024xf32>       // numpy reference
  %y = check.generate.fill value(0.0) : tensor<4x1024xf32>
  kernel.launch @g_mv_q6_k_k1024_t4[%n, %t](%n, %t, %acc, %x, %w, %y) : [index, index](index, index, i32, tensor<4x1024xf32>, tensor<860160xi8>, tensor<4x1024xf32>)
  check.expect.close actual(%y) expected(%e) atol(2.0e-3) rtol(2.0e-3) nan(different) : tensor<4x1024xf32>
  check.return
}
```
- Concatenate the kernel library and the cases into one file, then run
  `loom-test file.loom-test --device=amdgpu > out.json`. Each sample in
  `out.json["samples"]` has `case`, `passed` and `expectations.failures[].detail`
  ("element at index 8 (0) is not close to …").
- `check.generate.iota offset(a) step(b)` gives synthetic inputs. To A/B a new kernel
  against the old one, launch both on identical inputs and use `check.expect.close`.
  This is how every optimization was verified before the end-to-end golden run.
- For references, use `gguf.quants.dequantize(np.asarray(t.data), t.tensor_type)`;
  shapes are reversed (`t.shape = [K, N]`).
- `loom-run file.loom --root=@kernel --device=amdgpu --kernel-input-buffer=…` runs a
  single kernel. `loom-compile --root=@kernel` compiles one kernel; command programs
  can't be roots.
