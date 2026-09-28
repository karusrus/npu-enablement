# Results (branch hexagon-int-hmx; numbers measured at 4aa1a98, re-checked at ea11b8f)

Motorola Edge 70, Snapdragon 7 Gen 4 (SM7750), Android 16 (LineageOS), 12 GB. llama-bench, -r 3.

pp512, t/s — NPU: `-ngl 99 --device HTP0` (from 0f69bd6 no env vars; earlier builds needed `GGML_HEXAGON_HMX_Q40_ONLY=1 GGML_HEXAGON_FA_SELECT=1 GGML_HEXAGON_GDN_SELECT=1`);
CPU: `-ngl 0 --device none`, best of -t 2/4/6/8 (cpu-threads.txt); GPU: `-ngl 99 --device GPUOpenCL` (gpu-and-cpu8.txt).

| Model | NPU | CPU | GPU |
|---|---|---|---|
| Qwen3.5-4B Q4_0, all layers q4_0 | 171.7 | 51.7 | 73.9 |
| Llama-3.2-1B-Instruct Q4_0 | 551.5 | 226.8 | 307.2 |
| Qwen3.5-4B Q4_0 (unsloth mix) | 105.9 | 47.1 | 76.2 |

Perplexity, wikitext-2 test, `-c 512 --chunks 20 -b 512`, Qwen3.5-4B Q4_0 all layers q4_0:
CPU 10.8533 ± 0.42418 (4:02), NPU 10.8355 ± 0.42330 (1:08). BF16 reference 10.1409.

test-backend-ops -b HTP0: MUL_MAT (type_a=q4_0) 38/38; GATED_DELTA_NET 36/36; FLASH_ATTN_EXT 2583/2588 — the 5 failures are
kv = 8192–16384 cases with ERR 1e-3…6e-3 vs 5e-4, the same set as with GGML_HEXAGON_NHMX=0.

Greedy 30-token completion ("The capital of France is Paris. The capital of Germany is") is byte-identical
between NPU and CPU for Llama-3.2-1B and Qwen3.5-4B (cmp-*.txt).

Earlier builds (older numbers in some logs here): v1 pp512 Qwen 118, Llama 456; v1.1 126 / 521; v1.2 148 / 548.

From 0f69bd6 (HMX FP16 probe): on SM7750 without any env vars MUL_MAT 759/759 (all weight types), GATED_DELTA_NET 36/36,
FLASH_ATTN_EXT 2583/2588, PPL 10.8355, outputs identical to CPU. After cleanup (ea11b8f), interleaved A/B pp512 on
Qwen3.5-4B pure Q4_0: 172.0 before vs 173.1 after (3 runs each).

## v1.4 (28.09.2026, branch hexagon-int-hmx-pr, ff61a07 on a97cce8)

Short-row CONCAT (dim 0) and CPY go through VTCM with vgather. Qwen3.5-4B Q4_0 all layers q4_0, `-ngl 99 --device HTP0 -t 4`:

- tg64: 8.31 -> 8.99 t/s. Llama-3.2-1B tg64 30.1 t/s (no conv state, unchanged within noise).
- Per decode token (GGML_HEXAGON_PROFILE=1): CONCAT 5598 -> 656 us, CPY 2557 -> 518 us (24 layers).
- pp512 171.5 (unchanged). CPU on the same build: pp512 51.7 / 41.5 / 52.4 at -t 4/6/8 (NPU 172.0, 3.29x); tg64 9.68 (-t 4).
- test-backend-ops -b HTP0: CONCAT 48/48, CPY 136/136, GATED_DELTA_NET 36/36, SSM_CONV 45/45, MUL_MAT 759/759.
- Greedy 48 tokens after a 1442-token wikitext prompt: byte-identical to the CPU and to the NPU before the change.

Decode is memory-bound: with NPU decode running in the background, CPU decode drops from 9.7 to 4.1 t/s and NPU decode to ~7.5 t/s.

