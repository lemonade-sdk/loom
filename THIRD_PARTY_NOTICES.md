# Third-party material

- `skills/loom-benchmark/data/doc_*.txt` are concatenations of the Loom
  documentation (`loom/docs/src/**/*.md`) from
  [ROCm/hrx-system](https://github.com/ROCm/hrx-system), licensed under
  Apache-2.0. They are excerpts used as real-text benchmark prompts.
- `skills/loom-demo-ui/ui/short.wav` is the moonshine test recording
  (`model/moonshine-ai/moonshine-streaming-tiny/test/short.pcm`) from
  [ROCm/hrx-loom-library](https://github.com/ROCm/hrx-loom-library), licensed under
  MIT (Copyright (c) 2026 AMD ROCm Software), wrapped in a WAV header.
- `skills/loom-port-llm/starter/templates/*.loom` and parts of the generated kernels
  (Q4_K/Q6_K decode helpers, the text-service glue) are adapted from
  ROCm/hrx-loom-library (MIT), in particular `program/text/qwen3_8/` and
  `motif/format/ggml/`.
