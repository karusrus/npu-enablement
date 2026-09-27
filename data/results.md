# Results (branch hexagon-int-hmx @ 76074e0)

Motorola Edge 70, Snapdragon 7 Gen 4 (SM7750), Android 16 (LineageOS), 12 GB. llama-bench, -r 3.

pp512, t/s — NPU: `GGML_HEXAGON_HMX_Q40_ONLY=1 GGML_HEXAGON_FA_SELECT=1 GGML_HEXAGON_GDN_SELECT=1 -ngl 99 --device HTP0`;
CPU: `-ngl 0 --device none`, best of -t 2/4/6/8 (cpu-threads.txt); GPU: `-ngl 99 --device GPUOpenCL` (gpu-and-cpu8.txt).

| Model | NPU | CPU | GPU |
|---|---|---|---|
| Qwen3.5-4B Q4_0, all layers q4_0 | 147.7 | 51.7 | 73.9 |
| Llama-3.2-1B-Instruct Q4_0 | 547.7 | 226.8 | 307.2 |
| Qwen3.5-4B Q4_0 (unsloth mix) | 96.0 | 47.1 | 76.2 |

Perplexity, wikitext-2 test, `-c 512 --chunks 20 -b 512`, Qwen3.5-4B Q4_0 all layers q4_0:
CPU 10.8533 ± 0.42418 (4:02), NPU 10.8354 ± 0.42333 (1:18). BF16 reference 10.1409.

test-backend-ops -b HTP0: MUL_MAT (type_a=q4_0) 38/38; FLASH_ATTN_EXT 2583/2588 — the 5 failures are
kv = 8192–16384 cases with ERR 1e-3…6e-3 vs 5e-4, the same set as with GGML_HEXAGON_NHMX=0.

Greedy 30-token completion ("The capital of France is Paris. The capital of Germany is") is byte-identical
between NPU and CPU for Llama-3.2-1B and Qwen3.5-4B (cmp-*.txt).

Earlier builds (older numbers in some logs here): v1 pp512 Qwen 118, Llama 456; v1.1 126 / 521.
