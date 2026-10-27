# How loom-serve drives a text model (what the glue does)

The upstream reference is hrx-system `loom/docs/src/serving/model-programs.md`;
library examples live in `hrx-loom-library/program/text/qwen3_8/`.

## Package card (`model.json`)

- `programs.source.inputs`: `prepare.loom`, with `weights.loom` as its data input,
  plus `control.loom`, `storage.loom` and `service.loom`.
- `programs.source.libraries`: the kernel library and command program (`llm.loom`).
  Libraries compile per stage with stage constants applied.
- `variants.<v>.services.chat.parameters.model.files`: `[{path, format: "gguf"}]`.
  Weights are read in place (a symlink is fine); tensors are addressed by GGUF
  name in `command.parameter`.
- `variants.<v>.services.chat.config`: `{"llm.card": true}` selects scoped parameter
  sourcing in `prepare`.
- `presets.<p>.execution`:
  - `rows`: concurrent sequences;
  - `context_capacity`: maximum tokens per sequence;
  - `pool_capacity`: total KV positions, shared by all rows and the prefix cache;
  - `prefill_capacity`: tokens per prefill chunk. Shapes are powers of two up to
    this value, and the chunk equals prefill_capacity;
  - `max_tokens`: default completion length.

## Lifecycle

1. **prepare** (`@prepare`) runs once at load. It receives the packing shapes,
   each a `(tokens, spans)` pair, and creates one *stage* per shape. Each stage
   compiles `@llm_advance` with `llm.tokens`, `llm.spans`, `llm.context` and
   `llm.pool` set as constants, then binds the GGUF as parameter 0. It returns
   the effective prefill capacity, the stage count, the control state, the
   terminal marker string, and the storage records from `@llm_storage`.
2. **storage** (`@llm_storage`) declares the allocations:
   - slot 0: residual;
   - slot 1: arena, holding the KV pool followed by per-row private regions;
   - slot 2: metadata;
   - slot 3: origins/table;
   - slot 4: tokens;
   - slot 5: outputs.

   It also describes the KV geometry: planes = 2·layers, plane_bytes = pool ×
   POS_BYTES, block = 16 positions. The runtime pages KV in 16-position blocks
   and keeps a per-row block map in `origins`; `g_kv_offset()` resolves
   (row, position, plane) to an arena byte offset.
3. **control** functions turn requests into tokens and tokens back into text:
   - `chat_begin`, `chat_message` and `chat_end` render the prompt text.
     `chat_end` returns the generation prompt and the initial output state.
   - `prepare_input` tokenizes with special-token matching.
   - `output_next` splits the growing response into channels:
     - kind 0: consumed marker;
     - kind 1: answer;
     - kind 2: reasoning.

     State 1 means inside `<think>`, and state 2 skips the whitespace after
     `</think>`.
   - `encode_epoch` and `publish_epoch` translate the runtime's span plan into the
     **metadata packet**:
     - a 16-byte header, whose word 0 is the span count;
     - then 9 i32 per span: `[0]` length, `[1]` start position, `[3]` row,
       `[4]` first input token index, `[6]` flags, where bit 0 means "emit an
       output token for this span".
   - `epoch` selects the stage compiled for this packing shape.
4. **advance** (`@llm_advance`, the command program) runs once per epoch with all
   spans packed into `t = llm.tokens` token slots:
   - `g_token_info` maps each token slot to (row, position, live, span) from the
     packet, writing 16 bytes per token into `%info`.
   - Embed, the layers, and the head follow. The head computes the argmax for
     each selected span's last token and writes `outputs[span]`.
     `g_head_select` maps any `extra_eos_ids` to the EOS id.
   - Padding slots have `live = 0`. Kernels must not write KV for them, and
     GEMMs can skip tiles whose row scale is 0 (`GEMM_SKIP_PADDING`).

## Decode vs prefill inside one program

The same `@llm_advance` is specialized per shape, and `t` is a constant in each
stage:
- `t < 64` (decode, small batches): matvec kernels, using the 1-token kernel when
  `t == 1` and the 4-token one otherwise, plus split-K attention.
- `t ≥ 64` (prefill): f16 row-scaled WMMA GEMMs, with 128-row tiles from 128
  tokens. Attention uses flash2 on uniform 16-token tiles (one sequence,
  consecutive positions), and the listed split-K fallback handles the tokens of
  mixed tiles. `g_attention_fallback_list` builds that list.

## Tokenizer gotchas

- Use the HF `tokenizer.json` of the same model. GGUF-only repos usually ship
  without one, so download it from the base repo.
- loom-serve strips *special* added tokens from decoded text. If your output
  parser needs to see a structural token (e.g. Glimmer/ATEM `<|start|>`,
  `<|message|>`), mark it `"special": false` in the package tokenizer
  (`chat.nonspecial_tokens`). `<think>`/`</think>` are already non-special in
  Qwen3.
- `prepare_input` encodes with special-token matching, so special tokens in the
  rendered prompt become single ids, matching HF `apply_chat_template` + encode.

## ATEM/channel formats (non-ChatML)

The Glimmer port's control program (`examples/atem_channel_control.loom`) is
a full example of a channel-based format. Its prompt looks like
`<|start|>role<|message|>…<|eot|>`; reasoning is addressed `to=self`, and the
answer `to=user`. Its control program parses recipient headers in `output_next`.
Its benchmark helper rebuilds the raw output text from the channels the same way,
which is how `bench_common.raw_from_chat` must be extended for such formats.
