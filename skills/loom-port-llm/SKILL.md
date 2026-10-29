---
name: loom-port-llm
description: Convert an arbitrary LLM (GGUF checkpoint) into a loom-serve package — generate Loom IR kernels + command program from the GGUF's metadata, derive the chat/tool-calling glue from the model's own chat template, build model.json at the model's native context length, bring it up, unit-test kernels, and validate token-by-token against llama.cpp (short prompts, agent tool loops, full context). Includes a config-driven starter kit for dense transformers (Llama/Qwen2/Qwen3/Mistral-style, GQA, QK-norm, sliding window, output gates, post-norms, head_dim 128/256, partial RoPE) and Qwen3.5/Qwen3-Next-style hybrids (gated delta-net layers with recurrent state), with fast decode and WMMA prefill. Use for "port model X to Loom", "get X working on loom-serve", custom/unsupported models.
---

# Port an LLM to Loom

The goal is a package directory that `loom-serve --model=PKG/model.json` serves as a
complete chat model:
- greedy output that matches llama.cpp up to near-ties;
- the model's **native context length** (e.g. 40,960 for Qwen3-0.6B, 262,144 for
  Qwen3.5), with over-length prompts refused cleanly;
- **tool calling** in the model's own format, tool results, reasoning in history,
  the thinking on/off switch, and **history reuse** between agent turns.

The starter kit (`starter/`) gets a dense GGUF LLM, or a Qwen3.5-style hybrid, there
in minutes. Other architectures (MoE, MLA, other recurrent layers) need new kernels;
see "Beyond the starter" below.

Prerequisites:
- `loom-setup` is done: `~/loom/.venv` has the SDK (`hrx-loom` 0.1.0 from PyPI, which
  the templates target) and `~/loom/.tools` has gguf, numpy, tokenizers, jinja2 and
  requests.
- The llama.cpp ROCm reference is available (`loom-benchmark`), for the golden
  comparison.

## Workflow (Qwen3-0.6B and Qwen3.5-0.8B each take ~10 min of commands)

```bash
K=${CLAUDE_SKILL_DIR}/starter; T=~/loom/.tools/bin/python
W=~/work/mymodel && mkdir -p $W && cd $W
# 1. Inspect the GGUF: geometry, tensor formats, eos ids, native context, hybrid layers, and the
#    chat block DERIVED from the GGUF's Jinja chat template (chat_probe.py) and VERIFIED against it:
#    it prints "chat OK <conversation>" per probe (chat, multi-turn, tools, system+tools, agent loop,
#    parallel calls, with thinking on/off). Every line must be OK; a FAIL or "not derivable" means
#    fix cfg.json["chat"] by hand (fields: chat_probe.py) and rerun `chat_probe.py MODEL.gguf`.
$T -I $K/inspect_gguf.py /path/model-Q4_K_M.gguf cfg.json
# 2. REVIEW cfg.json against llama.cpp's src/models/<arch>.cpp - see the checklist below
# 3. Tokenizer: the HF tokenizer.json of the *same* model (base repo; GGUF repos often lack it)
~/loom/.venv/bin/hf download <org>/<base-model> tokenizer.json --local-dir tok
# 4. Build: context defaults to the model's native context, the KV pool to 2x that, both fitted to
#    --memory-gib (default 56; lower it if the machine is busy). Prints the model id and --memory_bytes.
$T $K/build.py cfg.json pkg --tokenizer tok/tokenizer.json --model-id <org>/<name-without-spaces>
# 5. Serve + smoke test
VENV=~/loom/.venv ${CLAUDE_SKILL_DIR}/../loom-setup/scripts/serve.sh mine 8160 amdgpu <memory_bytes> pkg/model.json
curl -s localhost:8160/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"<model id>","messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":400}'
#    thinking models spend 100-400 tokens reasoning: check finish_reason and that content is non-empty;
#    "chat_template_kwargs": {"enable_thinking": false} (or "reasoning_effort": "none") turns thinking off
# 6. Kernel unit tests vs numpy dequant references (embed, matvecs, RoPE, the whole attention chain,
#    the output head, and for hybrids the gated delta-net chain). Run them even if the output looks fine.
$T -I $K/make_kernel_tests.py cfg.json ktest && VENV=~/loom/.venv $K/run_kernel_tests.sh pkg ktest
# 7. Gates vs llama.cpp - loom-benchmark skill: golden.py/compare.py (short prompts + 1k-32k documents),
#    agent_golden.py (multi-turn tool calling, rendering, history reuse) and needle.py (full context).
```
After every rebuild, restart the server with `stop.sh` and then `serve.sh`; the
server does not reload the package (`serve.sh` stops a server whose registration failed).

**A port is done when** the kernel tests pass, the golden gate shows only near-tie
divergences, `agent_golden.py` passes (exact rendering, reuse inside tool loops,
valid calls), and `needle.py` passes up to the context capacity with a clean 400 above it.

## What the starter generates

| File | Role |
|---|---|
| `inspect_gguf.py` | GGUF metadata → `cfg.json`: hidden, layers, heads/kv_heads, head_dim, ffn, vocab, **context_length**, rms_eps, rope_theta/type, **rope_dims** (partial RoPE), qk_norm, attn_gate (`true`: separate tensor; `"fused_q"`: packed in attn_q), post_norms, sliding window/layers, NoPE layers, tied embeddings, eos ids, **recurrent_layers / gdn** (hybrids), the chat block (via `chat_probe.py`), and every tensor's format and shape. |
| `chat_probe.py` | Renders the model's Jinja chat template on probe conversations and cuts out what the control program needs: role prefixes/suffix, BOS, default system prompt, generation prompts with thinking on/off, the tools block and its order relative to the system message, the tool-call format (`hermes` JSON, `qwen_xml`, `glm`) and separators, tool-response framing, reasoning-in-history policy, trimming. `verify()` re-renders a dozen conversations exactly as `control.loom` will (`simulate()`) and diffs them against the template. `simulate()` is also what the benchmark scripts use to build golden prompts. |
| `gen_llm.py` | `cfg.json` → `llm.loom`: all kernels plus the `@llm_advance` command program. The command program is a **static per-layer unroll**, so each layer's kind (attention or gated delta-net) and tensor formats come from the GGUF (Q4_K_M mixes Q4_K/Q5_K/Q6_K/Q8_0 per tensor). |
| `gen_gdn.py` | Gated delta-net kernels (conv1d + SiLU, history commit, q/k L2 norms, register recurrence, gated RMSNorm) and the fused q/gate split, for hybrids. Recurrent state lives in the runtime's recurrent slots (see porting-notes). |
| `gen_gemm.py` | WMMA prefill GEMMs per (format, mode). Formats: q4_k, q5_k, q6_k, q8_0. Modes: store, acc, swiglu. |
| `gen_flash2.py` | Prefill flash attention (transposed, K/V staged once per KV head), head_dim 128 or 256. |
| `tile_uniform.loom` | Helper that classifies 16-token tiles for flash2. |
| `templates/*.loom` | Serving glue for the hrx-loom 0.1.0 text ABI (`text_*_v1` exports). `prepare` stages one program per packing shape. `storage` sets the KV geometry (2 planes per attention layer, F16) and the recurrent slot size. `control` is the chat policy: rendering, tools, reasoning, thinking switch, tokenization, output channels, tool-call parsing, canonical history for reuse. `service` maps the ABI exports. `weights.loom` is the parameter policy. |
| `build.py` | Fills the templates from `cfg.json["chat"]`, sizes context/pool/memory, then writes `model.json`, `tokenizer.json`, the GGUF symlink and `llm_config.json` (read by the benchmark scripts). |
| `make_kernel_tests.py`, `run_kernel_tests.sh` | `check.case` unit tests with numpy references (gguf dequantize) on the real tensor shapes: embed, matvecs (every format/K in use), QK-norm + RoPE, the attention chain (q/gate split → RoPE + paged KV store → split-K attention → gated combine), the output head argmax, and the gated delta-net chain on a span that continues a stored state. |

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
- head_dim 128 or 256 (attention); gated delta-net head dims multiples of 32;
- K-quant / Q8_0 weights (add dequant helpers for other formats); f32 norms and conv weights;
- hidden ≤ 8192; context ≤ 262,144 (kernel bounds; raise the `range(..., 262144)` facts for 1M-context models);
- **sampling**: `temperature`, `top_p`, `top_k` and `seed` per request. Temperature 0 (the default
  unless `build.py --temperature/--top-p/--top-k` set other defaults) is exact greedy. Otherwise the
  head draws from the top-k candidates (k ≤ 64; `top_k` 0 = 64) after temperature softmax and top-p,
  with a counter-based RNG keyed by (seed, position): a fixed `seed` reproduces a completion, and a
  request without one draws from its own nonce. Thinking models want sampling (Qwen: temperature
  0.6, top_p 0.95, top_k 20 for thinking; 0.7/0.8/20 without); greedy can end reasoning with EOS or
  loop, which `agent_golden.py` flags. Gates and goldens always send temperature 0.
- text only (image content parts are rejected; text parts are joined).

**Chat policy facts (control.loom):**
- Tools follow the template exactly (verified by `chat_probe.py`). Tool definitions are printed as
  the request sent them; send them serialized like Jinja `tojson` (`json.dumps` defaults) for
  byte parity with the template.
- Hermes-JSON calls are parsed in source without a JSON parser, so a malformed call never fails
  the request: its arguments go back as generated for the harness to handle. Qwen-XML and GLM
  calls go through the host `tools.parse_xml` codec (a malformed XML frame fails the request; a
  frame cut off by `max_tokens` is dropped). Text after the last complete call is dropped.
- Reasoning is rendered for assistant turns after the last user message, as the Qwen templates do.
  A new user turn therefore re-renders earlier turns without reasoning, and history reuse resumes
  from the first changed byte; inside a tool loop the previous turn is reused entirely
  (`prompt_tokens_details.cached_tokens`).
- loom-serve rejects `"reasoning_content": null` in a message: omit the field instead.
- `usage.prompt_tokens` of a turn that reuses history is one lower than `/v1/chat/completions/input_tokens`
  (the previous turn's EOS is already in the row); compare renderings with `input_tokens`.

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
- `chat`: derived and verified automatically (all `chat OK` lines). It matches the
  template byte for byte with **one deliberate exception**: when the model emits
  `<think>` itself (Qwen3 hybrid), `<think>\n` is appended to the thinking generation
  prompt so output starts in the reasoning channel. llama.cpp's *chat* endpoint then
  sees fewer prompt tokens than Loom; the goldens render with Loom's prompt, so they
  are exact. Check the printed summary against the model card:
  - `thinking`: `toggle` (Qwen3, Qwen3.5: `enable_thinking`), `always`, or `never`
    (`*-Instruct-2507`); `thinking_default` (Qwen3: on; Qwen3.5 small models: **off**);
  - `call_format`: `hermes` (Qwen2.5/Qwen3), `qwen_xml` (Qwen3.5/Qwen3-Coder), `glm`;
  - `default_system` (Qwen2.5 inserts "You are Qwen, created by Alibaba Cloud…").

  Templates the probe can't model (e.g. role tags with no closing suffix, tools
  inside the first user message, channel formats): fill the block by hand, keep
  `simulate()` in step, and rerun `chat_probe.py` until every line is OK. If the
  control program must parse structural tokens out of the generated text (e.g.
  channel tags), list them in `chat.nonspecial_tokens`: loom-serve strips special
  tokens from decoded text.
- Hybrids (`recurrent_layers` present): check `gdn` against llama.cpp's
  `src/models/qwen35.cpp` (k/v heads, head dims, conv kernel), that `kv_layers`
  counts only the attention layers, `rope_dims` (Qwen3.5: 64 of 256), `attn_gate:
  "fused_q"`, and `ffn_norm_tensor: "post_attention_norm"`.

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
   the KV pool (pool × 2·kv_layers × kv_heads·head_dim·2 bytes) plus recurrent state.
6. **Registration fails with a VGPR spill (`spill-traffic-register-exhausted`).** A
   kernel holds too much in registers for this model's shape. Reload operands per
   iteration instead of keeping them live (flash2 does this for head_dim 256).
7. **Coherent but wrong from the first token, kernel tests pass.** A wiring or
   config bug: read the generated command program for one layer of each kind
   (`awk '/---- layer N$/,/---- layer N+1/' pkg/program/llm.loom`). A real example:
   a stale cfg.json with `qk_norm: false` left Qwen3.5's Q/K norms out (`%no_norm`).
   An independent numpy forward of a short prompt that matches llama.cpp's top
   logits narrows "my understanding of the model" vs "my kernels" quickly; it is a
   diagnostic, never the golden.

## Beyond the starter (MoE, other formats, new ops)

- **MoE.** See `${CLAUDE_SKILL_DIR}/../loom-optimize/reference/flint-moe-example/`. It
  holds the generators from the RecursiveSI/flint port: a 40B MoE with sinks and
  sliding windows, about 1.6× llama.cpp decode. The pattern:
  1. router top-k;
  2. bucket tokens by expert, using an info buffer;
  3. grouped expert GEMMs over the buckets;
  4. combine.
- **Gemma-3n / Gemma-4 family (arch `gemma3n`, `gemma4`).** Read llama.cpp's
  `src/models/gemma*.cpp` closely. The starter has none of these features, and each
  one silently breaks output if missed:
  - **Per-layer embeddings (PLE).** A gathered `per_layer_token_embd` row, plus a
    projection of the hidden state, then norm, then a per-layer gate, projection
    and post-norm added to the residual. Count only the row read for bandwidth.
  - **KV sharing.** Layers ≥ `n_layer_kv_from_start` reuse the KV of the last
    owning layer of the same type (sliding or global), so they skip k/v and store
    nothing. Size the KV planes for the owning layers only.
  - **Mixed head_dim.** Sliding and global layers can differ (256 vs 512), with
    different rope theta.
  - **Partial "proportional" rope.** Global layers rotate only the pairs whose
    `rope_freqs` entry is finite (the rest are 1e30).
  - **Norms.** Q/K RMSNorm with weights, V RMSNorm *without* weights, attention
    scale 1.0, post-attention and post-FFN norms, and a `layer_output_scale`.
  - **FFN, embedding and head.** GELU-tanh gated FFN; embedding × √hidden; a final
    logit softcap, which doesn't affect argmax.
  - **Tensors.** bf16 tensors need a bf16 value helper (`view<32xbf16>` →
    `vector.extf`).

  Gemma's 512-wide global heads exceed the starter's head_dim 128/256 kernels: generalize
  the split-K attention (D/32 values per lane) and, for prefill, flash2 (it already loops
  over D/16 fragments and reloads Q when registers run out). Record it as a lever in the roofline.
- **New quant format.** Add `@g_<fmt>_values` and `@g_<fmt>_dot` funcs in `COMMON`
  (gen_llm.py), add the entries to `FMT`/`SB`, and add a `dequant16` case in
  `gen_gemm.py`. Unit-test against `gguf.quants.dequantize`.
- **Other recurrent layers (Mamba, RWKV, other delta-net variants).** Follow
  `gen_gdn.py`: per-span kernels read the reader slot at the fork position (else the
  row's writer slot), start from zeros at position 0, and publish the final state to
  the writer slot; `build.py` sizes the slot (`gdn_geometry()["slot_bytes"]`). Test the
  chain like `make_kernel_tests.py`'s `gdn_chain` (a span that continues a stored state).
- **Library programs are the canonical reference** for the text-service ABI (prepare,
  control, storage, service entry points). `templates/` follow the upstream tiny-text
  fixture (`loom/src/loom/serve/fixture/text` at tag v0.1.0 of ROCm/hrx-system) and
  `library/program/text/qwen3_8/chat.loom`. The upstream doc is
  `loom/docs/src/serving/model-programs.md`. When a new hrx-loom release changes the
  ABI, diff that fixture between the old and new tags; the registration error names
  the missing export (e.g. `root symbol '@text_chat_reasoning_v1' was not found`).

## Reference

- `reference/loom-ir-cookbook.md`: IR idioms, proof facts, and compile-error fixes.
- `reference/porting-notes.md`: how the text service drives the program (epochs,
  spans, metadata packets, KV paging), and what each glue file does.
- Worked examples: `${CLAUDE_SKILL_DIR}/../loom-optimize/reference/flint-moe-example/`
  (MoE port sources), plus `reference/examples/atem_channel_control.loom` and
  `atem_render_and_reconstruct.py` (Glimmer's channel-based chat format, its prompt
  rendering, and how the raw output is rebuilt for golden comparison).
