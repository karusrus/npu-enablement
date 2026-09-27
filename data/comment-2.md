Thanks for checking on IQ9! I dug further on the device, and I think the root cause is a hardware capability difference, not a kernel bug: **the HMX on SM7750 appears to have no FP16 mode**.

**1. It's exactly the HMX paths.** Full `-o MUL_MAT` with `GGML_HEXAGON_VERBOSE=1`: all 87 tests that run `hmx-tiled` FAIL, all 672 that run `hvx-tiled` pass. Thread count (`NHVX=1..4`) and `OPPOLL` make no difference; the failure is deterministic.

**2. Resource setup looks fine.** `HAP_compute_res_hmx_lock()` returns 0, HMX power-on succeeds, and `HAP_compute_res_query_capability(PREEMPTION)` returns not-supported (legacy model, same as older firmware).

**3. HMX writes zeros.** I instrumented `hmx_matmul_worker_fn` for `MUL_MAT(q4_0, m=16, n=5, k=256)`. The activation and weight tiles in VTCM hold sane fp16 values, and the scales are `0x3c00`. I pre-filled the output tile with `0x5555`. After `core_dot_chunk_fp16` the tile is all `0x0000`, so the store executes but the accumulator is zero. With scale = 1.0 **and** bias = 1.0 in the bias block the output is still all zeros, so not even the bias gets applied.

**4. The product brief agrees.** Qualcomm's Snapdragon 7 Gen 4 brief lists the NPU as "Support for precisions INT4, INT8, and INT16". The 7+ Gen 3 brief says "Support for all precisions (INT4, INT8, INT16, FP16)". The chip still reports itself as v73 with `hmx 1`, so the backend enables the FP16 HMX kernels.

**5. The NPU itself works.** QNN HTP runs SD 1.5 correctly on this phone (Local Dream app, 512×512, 20 steps in 17 s). I assume that model is integer-quantized, since FP16 isn't listed for this chip.

**Workaround:** `GGML_HEXAGON_NHMX=0` gives MUL_MAT 759/759 and GATED_DELTA_NET 36/36 passing. FLASH_ATTN_EXT 2583/2588; the 5 failures are small precision misses on very long KV (kv=8192–16384, ERR 1e-3…5e-3 vs 5e-4), not garbage.

**Suggested fix:** the backend can't tell FP16-capable HMX from the SoC version alone. One option is a tiny self-test at session init: multiply one 32×32 tile of ones, expect 32.0, and on failure set `n_hmx = 0` with a warning. I'm happy to send a PR for that if the approach works for you, or to test whatever you prefer. Also for reference: VTCM here is 4 MB (hwinfo) vs 8 MB on your IQ9.

**Question:** is an integer HMX path (int8/int4 weights, int16 activations) on your roadmap for parts like this? On SM7750 that's the only way to use HMX for matmul, and I'd rather not duplicate work if you're already on it.

Disclosure: this debugging was done with Claude Code (AI); all numbers are from runs on this device.
