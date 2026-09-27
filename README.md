# Integer HMX for llama.cpp on Snapdragon 7 Gen 4

On Snapdragon 7 Gen 4 (SM7750, Hexagon v73), llama.cpp's Hexagon backend produces garbage:
the HMX matrix unit on this chip has no FP16 support, while ggml-hexagon enables FP16 HMX
based on the architecture version (v73+). Qualcomm's own runtime confirms it — QNN 2.50 on
SM7750 logs `Hexagon arch=73 ... fp16=false`.

This work moves the math to **integer HMX**, which the chip does support. Result: prompt processing
on the NPU is **2.4–2.9× faster than the best CPU configuration and 1.8–2× faster than the Adreno GPU**,
with the same perplexity and byte-identical outputs.

Code: branch [`hexagon-int-hmx`](https://github.com/karusrus/llama.cpp/tree/hexagon-int-hmx) in my llama.cpp fork
(one commit on top of upstream `86a24a1`). Upstream discussion: ggml-org/llama.cpp#29473. Work in progress, not upstream.

## Results

Motorola Edge 70 (Snapdragon 7 Gen 4, 12 GB), llama-bench `pp512`, tokens/s.
CPU = best of 2/4/6/8 threads. GPU = stock OpenCL backend (Adreno 722).

| Model | NPU (this patch) | CPU | GPU | vs CPU | vs GPU |
|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct Q4_0 | 547.5 | 226.8 | 307.2 | 2.41× | 1.78× |
| Qwen3.5-4B Q4_0 (all layers q4_0) | 147.1 | 51.7 | 73.9 | 2.85× | 1.99× |
| Qwen3.5-4B Q4_0 (unsloth, mixed quants) | 96.0 | 47.1 | 76.2 | 2.04× | 1.26× |

Quality, Qwen3.5-4B pure Q4_0, wikitext-2, 20 × 512 tokens:

| | Perplexity | Time |
|---|---|---|
| CPU | 10.8533 ± 0.424 | 4:02 |
| NPU | 10.8354 ± 0.423 | 1:18 |

Greedy outputs on the NPU are byte-identical to the CPU for all three models.
`test-backend-ops`: MUL_MAT q4_0 38/38, FLASH_ATTN_EXT 2583/2588 (5 long-context cases, kv 8K–16K, still fail).

## How it works

- **Weights.** q4_0 blocks are re-quantized to int8 per column against the maximum block scale of the
  current K chunk. That scale goes into the HMX output conversion, shifted so the 16-bit store cannot overflow.
- **Activations.** Stored as 16-bit and fed as two 8-bit planes (the `uh:2x1` mode).
  A second bias word cancels the unsigned offset exactly.
- **Accumulation.** HMX runs the integer matmul. Two stores per K chunk (coarse + fine) are stitched
  on HVX into the exact 32-bit sum, then rescaled per row and column.
- **Pipeline.** HMX computes tile *i* while HVX reduces tile *i−1* and converts weights for tile *i+1*.
- **Attention.** Q·Kᵀ and P·V both run on integer HMX. K is smoothed per channel, with the factor
  folded back into Q. Softmax runs on HVX in fp16, and the 1/sum factor is folded into the row scale.

The integer HMX behaviour was mapped by testing on the Hexagon SDK's x86 emulator, then verified on the
phone with the same inputs: 12/12 output hashes match.

## Which chips need this

Qualcomm AI Hub (QNN 2.50) reports HMX FP16 per chipset:

| FP16 | Chipsets |
|---|---|
| **no** | Snapdragon 888 (v68), 778G (v68), QCS6490 (v68), **7 Gen 4 (v73)**, **QCM6690 (v73)** |
| yes | 8 Gen 1 (v69), 8 Gen 2 (v73), 8 Gen 3 (v75), 8 Elite (v79), 8 Elite Gen 5 (v81), X Elite / X Plus (v73), X2 Elite (v81), QCS8550 / IQ-9075 (v73), QCS8275 (v75), SA8295P (v68), SA8775P (v73), SA7255P (v75) |

On v73 the backend enables FP16 HMX, so 7 Gen 4 and QCM6690 are exposed. 7 Gen 4 is confirmed on a real
device; QCM6690 is untested.

## Limits

- Prefill only. Decode (tg128, t/s) is memory-bound and the CPU stays fastest:
  Llama-3.2-1B — CPU 33.5, NPU 29.5, GPU 24.8; Qwen3.5-4B — CPU 9.8, GPU 8.2, NPU 8.0.
- Splitting one model's layers across NPU + GPU is slower than the NPU alone: 364–397 vs 456 t/s (same build) on
  Llama-3.2-1B pp512. Layers run one after another, so the speed lands between the two devices.
- Small batches are slower than the CPU: 8 tokens take 336 ms on the NPU vs 203 ms on the CPU (Qwen3.5-4B).
  The weight conversion costs the same for 8 tokens as for 512.
- Only q4_0 weights run on HMX. Other types stay on CPU/HVX, which is why mixed-quant models gain less.
- Tested on one device.
- Headroom: HMX is idle most of the time; HVX weight conversion and output reduction are the bottleneck.
  Qualcomm's QNN reaches ~9.6 TFLOPS (w4a16) on the same chip.

## Build and run

Same as upstream llama.cpp for Snapdragon (docs/backend/snapdragon), on the branch:

```
git clone -b hexagon-int-hmx https://github.com/karusrus/llama.cpp
cd llama.cpp
docker run -it --rm -u $(id -u):$(id -g) --volume $(pwd):/workspace --platform linux/amd64 \
  ghcr.io/snapdragon-toolchain/arm64-android:v0.7
# inside the container
cp docs/backend/snapdragon/CMakeUserPresets.json .
cmake --preset arm64-android-snapdragon-release -B build-snapdragon
cmake --build build-snapdragon
cmake --install build-snapdragon --prefix pkg-android/llama.cpp
```

On the phone (SM7750):

```
GGML_HEXAGON_HMX_Q40_ONLY=1 GGML_HEXAGON_FA_SELECT=1 GGML_HEXAGON_GDN_SELECT=1 \
  ./bin/llama-bench -m model.gguf -p 512 -n 0 -ngl 99 --device HTP0
```

- `GGML_HEXAGON_HMX_Q40_ONLY=1` sends only plain Q4_0 matmuls to HMX (they take the integer path); other types stay on HVX.
- `GGML_HEXAGON_FA_SELECT=1`, `GGML_HEXAGON_GDN_SELECT=1` keep attention and gated delta net off the stock FP16 HMX
  kernels; attention prefill then takes the integer HMX path.
- Use a model where all matmul weights are Q4_0 (`llama-quantize --pure ... Q4_0`) for the full speedup.

## Repo layout

`patches/` — the same change as the branch, as a patch (MIT, like llama.cpp) · `src/` — the integer HMX kernels ·
`aihub/` — FP16 chip map and QNN ceiling scripts · `scripts/` — benchmark scripts run on the phone · `data/` — raw logs.
