---
name: loom-port-llm
description: Convert an arbitrary LLM (GGUF checkpoint) into a loom-serve package — generate Loom IR kernels + command program from the GGUF's metadata, add chat/control glue, build model.json, bring it up, unit-test kernels, and validate token-by-token against llama.cpp. Includes a config-driven starter kit for dense transformers (Llama/Qwen2/Qwen3/Mistral-style, GQA, optional QK-norm, sliding window, output gate, post-norms) with fast decode and WMMA prefill. Use for "port model X to Loom", "get X working on loom-serve", custom/unsupported models.
---

# Port an LLM to Loom

The goal is a package directory that `loom-serve --model=PKG/model.json` serves, with
greedy output that matches llama.cpp up to near-ties. The starter kit
(`starter/`) gets a **dense GGUF LLM** there in minutes. Other architectures (MoE,
hybrid/recurrent, MLA) need new kernels; see "Beyond the starter" below.

Prerequisites:
- `loom-setup` is done: `~/loom/.venv` has the SDK and `~/loom/.tools` has gguf,
  numpy, tokenizers and requests.
- The llama.cpp ROCm reference is available (`loom-benchmark`), for the golden
  comparison.

## Workflow (Qwen3-0.6B took ~15 min end to end)

```bash
K=${CLAUDE_SKILL_DIR}/starter; T=~/loom/.tools/bin/python
W=~/work/mymodel && mkdir -p $W && cd $W
# 1. Inspect the GGUF: config with geometry, tensor formats, eos ids, chat block (ChatML is auto-detected)
$T -I $K/inspect_gguf.py /path/model-Q4_K_M.gguf cfg.json
# 2. REVIEW cfg.json against llama.cpp's src/models/<arch>.cpp (or src/llama-model.cpp) - see checklist below
# 3. Tokenizer: the HF tokenizer.json of the *same* model (base repo; GGUF repos often lack it)
~/loom/.venv/bin/hf download <org>/<base-model> tokenizer.json --local-dir tok
# 4. Build the package (generates kernels + glue + card); it prints the --memory_bytes to use
$T $K/build.py cfg.json pkg --tokenizer tok/tokenizer.json --context 32768 --pool 65536 --rows 4
# 5. Serve + smoke test
VENV=~/loom/.venv ${CLAUDE_SKILL_DIR}/../loom-setup/scripts/serve.sh mine 8160 amdgpu <memory_bytes> pkg/model.json
curl -s localhost:8160/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<cfg name>","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":400}'
#    thinking models spend 100-400 tokens reasoning: check finish_reason and that content is non-empty
# 6. If the output is garbage: kernel unit tests vs numpy dequant references
$T -I $K/make_kernel_tests.py cfg.json ktest && VENV=~/loom/.venv $K/run_kernel_tests.sh pkg ktest
# 7. Correctness gate vs llama.cpp (short prompts + 1k/2k/16k/32k documents) - loom-benchmark skill
```
After every rebuild, restart the server with `stop.sh` and then `serve.sh`; the
server does not reload the package.

## What the starter generates

| File | Role |
|---|---|
| `inspect_gguf.py` | GGUF metadata → `cfg.json`: hidden, layers, heads/kv_heads, head_dim, ffn, vocab, rms_eps, rope_theta/type, qk_norm, attn_gate, post_norms, sliding window/layers, NoPE layers, tied embeddings, eos ids, chat template block, and every tensor's format and shape. |
| `gen_llm.py` | `cfg.json` → `llm.loom`: all kernels plus the `@llm_advance` command program. The command program is a **static per-layer unroll**, so each layer's tensor formats come from the GGUF (Q4_K_M mixes Q4_K/Q6_K per layer). |
| `gen_gemm.py` | WMMA prefill GEMMs per (format, mode). Formats: q4_k, q6_k, q8_0. Modes: store, acc, swiglu. |
| `gen_flash2.py` | Prefill flash attention (transposed, K/V staged once per KV head). |
| `tile_uniform.loom` | Helper that classifies 16-token tiles for flash2. |
| `templates/*.loom` | Serving glue. `prepare` stages one program per packing shape. `storage` sets the KV geometry (2 planes per layer, F16). `control` handles chat rendering, tokenization and output parsing (reasoning/answer channels). `service` is the text-service ABI. `weights.loom` is the parameter policy. |
| `build.py` | Fills the templates from `cfg.json["chat"]`, then writes `model.json`, `tokenizer.json`, the GGUF symlink and `llm_config.json` (read by the benchmark scripts). |
| `make_kernel_tests.py`, `run_kernel_tests.sh` | `check.case` unit tests with numpy references (gguf dequantize) for the embed, matvec and rope kernels, run on the real tensor shapes. |

Kernel inventory:
- **Decode:**
  - one-lane-per-32-value-group matvecs, 1-token and 4-token variants, with fused SwiGLU;
  - an argmax output head that writes no logits to memory;
  - split-K decode attention with up to 64 splits and early exit.
- **Prefill:**
  - f16 row-scaled activations feeding WMMA GEMMs;
  - transposed flash attention;
  - a split-K fallback for tokens in mixed (multi-sequence) tiles.
- **Shared:** a per-token QK-norm + RoPE + F16 KV store kernel, and fused add + RMSNorm.

**Fixed assumptions:**
- head_dim = 128;
- K-quant / Q8_0 weights (add dequant helpers for other formats);
- hidden ≤ 8192;
- greedy decoding only (sampling fields are accepted and ignored);
- text only (no tools or images).

## cfg.json review checklist (step 2)

Compare with llama.cpp's graph builder for the arch (`src/models/<arch>.cpp` in
recent llama.cpp; older trees have it in `src/llama-model.cpp`). Check these fields:
- `rope_type`:
  - `neox` pairs (d, d+64). Used by qwen2/qwen3/gemma/phi3/…; llama.cpp's
    `LLAMA_ROPE_TYPE_NEOX`.
  - `norm` pairs (2d, 2d+1). Used by llama/mistral; `LLAMA_ROPE_TYPE_NORM`.
  - A wrong value still gives plausible-looking text for a few tokens, then degrades.
- `rope_theta`; rope scaling (YaRN/linear) is **not** implemented, so add it in
  `qk_rope()` if the model uses it.
- `qk_norm` (per-head RMSNorm on q/k before RoPE), `post_norms` (Gemma-style
  post-attention/FFN norms), `attn_gate` (sigmoid output gate), `embed_scale`
  (Gemma multiplies embeddings by √hidden), `embed_norm`.
- Sliding-window layers: list them in `sliding_layers` and set `sliding_window`.
  `nope_layers` are layers without RoPE.
- `extra_eos_ids`: every id that should stop generation. For ChatML models that is
  `<|im_end|>` and `<|endoftext|>`.
- `chat`: must match the model's Jinja template byte for byte, with **one
  deliberate exception**: thinking models get `<think>\n` appended to the
  generation prompt so output starts in the reasoning channel. The model would
  emit it anyway. As a result, llama.cpp's *chat* endpoint sees 2 fewer prompt
  tokens than Loom, while `golden.py` uses Loom's rendering, so the goldens are
  exact. Check these parts:
  - message prefixes and suffix;
  - the generation prompt;
  - whether generation starts inside `<think>` (`starts_in_reasoning`);
  - `think_close` (what follows reasoning in history);
  - `terminal`.

  `inspect_gguf.py` decides thinking mode from the template's
  `add_generation_prompt` block and prints it. Check it against the model card:
  - Hybrid Qwen3 (e.g. Qwen3-0.6B) thinks. Use `generation_prompt`
    `<|im_start|>assistant\n<think>\n` with `starts_in_reasoning: true`.
  - Instruct-only variants (e.g. `*-Instruct-2507`) never think. Use
    `<|im_start|>assistant\n` with `false`, even though their template mentions
    `<think>` to strip it from history.

  Non-ChatML templates: fill it by hand. If the control program must parse
  structural tokens out of the generated text (e.g. channel tags), list them in
  `chat.nonspecial_tokens`. loom-serve strips special tokens from decoded text.

## Bring-up debugging order

1. **The server fails to register.** `serve.sh` prints diagnostics. See
   `reference/loom-ir-cookbook.md` for each diagnostic family (PARSE, TYPE,
   SUBRANGE, TARGET, SYMBOL).
2. **Garbage from the first token.** Run the kernel unit tests. They found the
   one-wave matvec writer bug (K=1024 gives 32 lanes for 64 outputs) in seconds.
   Then suspect, in order:
   - rope pairing;
   - the norm eps;
   - a tied head that is missing `output.weight` (the head must use `token_embd.weight`);
   - a tensor format with no dequant helper.
3. **Correct short answers but wrong long ones.** The prefill path (≥64 tokens:
   GEMM + flash2 + fallback) is wrong, or KV paging is. Compare 1k/2k goldens
   against short goldens, and run `consistency.py` for mixed batches.
4. **Correct but stops early or never stops.** Check the EOS ids and the chat
   `terminal`.
5. **503 "insufficient completion capacity".** `--memory_bytes` is too small for
   the KV pool (pool × 2·layers × kv_heads·128·2 bytes).

## Beyond the starter (MoE, other formats, new ops)

- **MoE.** See `${CLAUDE_SKILL_DIR}/../loom-optimize/reference/flint-moe-example/`. It
  holds the generators from the RecursiveSI/flint port: a 40B MoE with sinks and
  sliding windows, about 1.6× llama.cpp decode. The pattern:
  1. router top-k;
  2. bucket tokens by expert, using an info buffer;
  3. grouped expert GEMMs over the buckets;
  4. combine.
- **New quant format.** Add `@g_<fmt>_values` and `@g_<fmt>_dot` funcs in `COMMON`
  (gen_llm.py), add the entries to `FMT`/`SB`, and add a `dequant16` case in
  `gen_gemm.py`. Unit-test against `gguf.quants.dequantize`.
- **Recurrent/hybrid (Mamba, gated delta-net).** The KV storage template only
  covers attention planes. Study the library's `program/text/qwen3_8/` (hybrid)
  storage/epoch programs.
- **Library programs are the canonical reference** for the text-service ABI (prepare,
  control, storage, service entry points). `templates/` came from them. The
  upstream doc is `loom/docs/src/serving/model-programs.md`.

## Reference

- `reference/loom-ir-cookbook.md`: IR idioms, proof facts, and compile-error fixes.
- `reference/porting-notes.md`: how the text service drives the program (epochs,
  spans, metadata packets, KV paging), and what each glue file does.
- Worked examples: `${CLAUDE_SKILL_DIR}/../loom-optimize/reference/flint-moe-example/`
  (MoE port sources), plus `reference/examples/atem_channel_control.loom` and
  `atem_render_and_reconstruct.py` (Glimmer's channel-based chat format, its prompt
  rendering, and how the raw output is rebuilt for golden comparison).
