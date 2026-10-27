# Calibration: achieved efficiency vs roofline (Strix Halo gfx1151)

Ceilings are from `roofline.py` (216 GB/s, 59 TFLOP/s). Achieved numbers are
end-to-end and client-side, from tuned Loom ports. `EFF` in `roofline.py` is the
best (or a representative) value per class. Add one row per finished optimization
campaign, including ones that missed their target.

| Model (arch, quant) | Metric | Ceiling | Loom achieved | Eff | llama.cpp ROCm | Note |
|---|---|---|---|---|---|---|
| Nemotron-3.5-Bolt-E4B (gemma4 dense, Q4_K_M) | decode | 68.1 | 61.1 | 0.90 | 52.7 | adaptive matvec rows, fused QKV, paired-row geglu |
| 〃 | decode @1k | 66.8 | 59.2 | 0.89 | 50.6 | |
| 〃 | decode @2k | 66.2 | 57.9 | 0.87 | 48.7 | |
| 〃 | prefill 1k | 7146 | 1830 | 0.26 | 1593 | head_dim 256/512: no flash2, simple split attention |
| 〃 | prefill 2k | 7094 | 1801 | 0.25 | 1678 | |
| 〃 | c4 steady | 255 | 165 | 0.65 | 157 | |
| Bolt-E4B, run 2 (fresh agent, 1 turn, 1 h budget) | decode | 68.1 | 60.0 | 0.88 | 52.7 (b1355) | adaptive rows only; fused QKV / paired-row not tried |
| 〃 | prefill 1k / 2k | 7146 / 7094 | 2201 / 2339 | 0.31 / 0.33 | 1593 / 1678 | new flash prefill kernel for head_dim 256/512 (~30 min to build) |
| 〃 | c4 steady | 255 | 169 | 0.66 | 157 | |
| LFM2.5-24B-A2B (lfm2moe hybrid: 30 short-conv + 10 attention layers, MoE 64×4, Q4_K_M; fresh agent, 1 turn, 74 min) | decode | 140 | 100.4 | 0.72 | 97.3 (b1355) | MoE decode below 0.78; the roofline step was skipped, so no targets steered the run |
| 〃 | prefill 1k / 2k | 13208 / 12970 | 2677 / 2744 | 0.20 / 0.21 | 2140 / 2213 | prefill chunk 4096: every chunk reads all 64 experts |
| 〃 | c4 steady | 204 | 185.7 | 0.91 | 158.9 | |
| Qwen2.5-0.5B-Instruct (qwen2 dense, Q4_K_M/Q5_0; fresh agent, 1 turn, 35 min, campaign tools used) | decode | 551 (target 377 with per-layer floor) | 289 | 0.52 | 288 (b1355) | small model: per-layer launch cost dominates |
| 〃 | prefill 1k / 2k / 16k | 77769 / 73596 / – | 17571 / 18571 / 11920 | 0.23 / 0.25 / 0.28 | 13851 / 13890 / 8202 | |
| Flint (MoE 256×6, int4 q4_1) | decode | 130 | 99–104 | 0.78 | 62 | expert gather bound |
| 〃 | decode @32k | 82.7 | 63 | 0.76 | 48 | |
| 〃 | prefill 2k | 11946 | ~2650 | 0.22 | 1509 | |
| 〃 | prefill 32k | 7041 | 1976 | 0.28 | 1047 | |
| 〃 | c4 steady | 244 | 181 | 0.74 | 152 | |

Observations:
- Dense decode reaches about 0.9 of the bandwidth roofline. The last 10 % is the
  per-layer small kernels plus attention.
- Prefill end to end is ~0.25–0.33 of WMMA peak, even though the GEMMs alone reach ~0.5.
  Attention, norms and dispatch make up the rest. A target above ~0.3 needs new
  GEMM work (LDS double-buffering, dequant into fragments).
- These efficiencies were fitted on the same two models, so targets for a new
  architecture are an out-of-sample prediction. Record the result here.
