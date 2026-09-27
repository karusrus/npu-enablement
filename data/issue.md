### Name and Version

```
$ LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib ./bin/llama-cli --version
ggml-hex: forcing ndev to 1 for SoCs archs lower than v75.
version: 0.5.0-dev (build 1, commit 86a24a1)
built with Clang 21.0.0 for Android aarch64
```

Built with the official toolchain image `ghcr.io/snapdragon-toolchain/arm64-android:v0.7`, preset `arm64-android-snapdragon-release` from `docs/backend/snapdragon/CMakeUserPresets.json`, installed to `/data/local/tmp/llama.cpp` and run from `adb shell` as described in `docs/backend/snapdragon/README.md`.

### Operating systems

Other? (Please let us know in description)

### GGML backends

Hexagon

### Hardware

Motorola Edge 70 (`roadstr`, XT2601-2), Qualcomm **Snapdragon 7 Gen 4 (SM7750)**, Hexagon **v73**, 12 GB LPDDR5X.

```
ggml-hex: Loading driver libcdsprpc.so
ggml-hex: forcing ndev to 1 for SoCs archs lower than v75.
ggml-hex: FASTRPC_GET_DOMAINS query failed (0x6c), using static CDSP domains
ggml-hex: Hexagon backend (experimental) : allocating new registry : ndev 1
ggml-hex: Hexagon Arch version v73, DMA64 disabled
ggml-hex: device 0: HTP0 (phys=0, virt=0, domain=cdsp:3)
```

OS: Android 16. Full disclosure: the phone runs an unofficial LineageOS 23.2 build, but vendor, DSP and firmware images are stock Motorola (`ro.vendor.build.fingerprint = motorola/roadstr_g/roadstr:16/W1WRS36.39-115-2/549c34:user/release-keys`). `ro.soc.model = SM7750`.

### Models

- `bartowski/Llama-3.2-1B-Instruct-GGUF` — `Llama-3.2-1B-Instruct-Q4_0.gguf`
- `unsloth/Qwen3.5-4B-GGUF` — `Qwen3.5-4B-Q4_0.gguf`

### Problem description & steps to reproduce

On SM7750 / v73 the Hexagon backend runs fast but produces garbled output for both models. The CPU backend on the same device, same binaries and same prompt answers correctly.

**Reproduce (greedy, same prompt):**

```
./bin/llama-completion -m models/Llama-3.2-1B-Instruct-Q4_0.gguf -t 4 -n 50 --temp 0 -no-cnv -p "$PROMPT" -dev none -ngl 0
  -> "I'm not aware of any Snapdragon phone chip that has three processors that can run a neural network. ..."

./bin/llama-completion -m models/Llama-3.2-1B-Instruct-Q4_0.gguf -t 4 -n 50 --temp 0 -no-cnv -p "$PROMPT" -dev HTP0 -ngl 99
  -> "lius (1872)). The Americans are indifferent to the best. However, don't have a brother (sister), 1968, Virginia, ..."
```

Qwen3.5-4B behaves the same way: correct on CPU, mixed-script garbage on `HTP0`.

**`test-backend-ops -b HTP0` narrows it down to three ops:**

1. **MUL_MAT through HMX fails for n >= 5** (n = 1 passes). With `GGML_HEXAGON_MM_SELECT=1` (HVX only) MUL_MAT passes: 761 OK, 0 FAIL. `GGML_HEXAGON_NHMX=1`, `GGML_HEXAGON_HOSTBUF=0` and `GGML_HEXAGON_VMEM=0` do not help (89 FAIL each).
   ```
   [MUL_MAT] ERR = inf > 0.000500000   MUL_MAT(type_a=q4_0,type_b=f32,m=16,n=5,k=256,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0): FAIL
   [MUL_MAT] ERR = inf > 0.000500000   MUL_MAT(type_a=q8_0,type_b=f32,m=6,n=4096,k=5120,bs=[1,1],nr=[1,1],per=[0,1,2,3],k_v=0,o=1,src_overlap=0): FAIL
   ```
2. **FLASH_ATTN_EXT** (1869 failing cases in the full run with `MM_SELECT=1`):
   ```
   [FLASH_ATTN_EXT] ERR = inf > 0.000500000   FLASH_ATTN_EXT(hsk=64,hsv=64,nh=8,nr23=[4,1],kv=1024,nb=32,mask=1,sinks=0,max_bias=0.000000,logit_softcap=0.000000,prec=f32,type_K=f16,type_V=f16,permute=[0,1,2,3],kv_view=1,v_is_view_of_k=0,n_kv_max=0): FAIL
   ```
3. **GATED_DELTA_NET**:
   ```
   [GATED_DELTA_NET] ERR = 10275358628165832.000000000 > 0.000000100   GATED_DELTA_NET(type=f32,head_count=4,head_size=64,n_seq_tokens=64,n_seqs=1,v_repeat=1,permuted=0,kda=0,K=1): FAIL
   [GATED_DELTA_NET] ERR = inf > 0.000000100   GATED_DELTA_NET(type=f32,head_count=4,head_size=64,n_seq_tokens=256,n_seqs=1,v_repeat=1,permuted=0,kda=0,K=1): FAIL
   ```

ADD / SUB / DIV also show a few f16 failures, but these are tiny precision deltas (ERR ~1.5e-7 vs 1e-7 threshold) and look harmless; MUL passes 67/67.

**Workaround that gives correct output:** `GGML_HEXAGON_MM_SELECT=1 GGML_HEXAGON_OPFILTER='^(FLASH_ATTN_EXT|GATED_DELTA_NET)$'` plus `-fa off`. Output then matches the CPU backend for both models, but without HMX the NPU is slower than the CPU, so the backend currently brings no benefit on this SoC:

| Llama-3.2-1B Q4_0, `-t 4` | pp512 t/s | tg64 t/s | output |
|---|---|---|---|
| CPU | 216.3 | 34.4 | correct |
| HTP0, default (HMX) | 1749.4 | 29.9 | garbled |
| HTP0, workaround above | 78.7 | 27.0 | correct |

| Qwen3.5-4B Q4_0, `-t 4` | pp512 t/s | tg64 t/s | output |
|---|---|---|---|
| CPU | 46.3 | 9.1 | correct |
| GPU (OpenCL, Adreno 722) | 77.5 | 7.5 | correct |
| HTP0, default (HMX) | 437.4 | 8.1 | garbled |
| HTP0, workaround above | 25.6 | 6.2 | correct |

My guess, not verified: the HMX / FA / GDN kernels assume something about v73 that holds on Snapdragon 8 Gen 2 but not on the 7-series part (VTCM size or HMX configuration?). Similar reports for other SoCs were #25876 (SM8850, v81) and #24365 (SM8845, v81).

I have the toolchain set up and can rebuild and test patches or run extra diagnostics on this device.

### First Bad Commit

Unknown — first build of the Hexagon backend on this device was at `86a24a1`.

### Relevant log output

```
$ GGML_HEXAGON_MM_SELECT=1 ./bin/test-backend-ops -b HTP0   (failures grouped by op)
   1869 FLASH_ATTN_EXT
     33 DIV        (f16, ERR ~1e-7)
     14 ADD        (f16, ERR ~1e-7)
     12 MUL        (full run only; `-o MUL` alone passes 67/67)
      7 GATED_DELTA_NET
      1 SUB        (f16, ERR ~1e-7)
```
