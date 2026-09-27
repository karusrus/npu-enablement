cd /data/local/tmp/llama.cpp
export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib
for M in Llama-3.2-1B-Instruct-Q4_0 Qwen3.5-4B-Q4_0-pure Qwen3.5-4B-Q4_0; do
  N=$(GGML_HEXAGON_HMX_Q40_ONLY=1 GGML_HEXAGON_FA_SELECT=1 GGML_HEXAGON_GDN_SELECT=1 timeout 900 ./bin/llama-bench -m models/$M.gguf -p 512 -n 0 -r 3 -ngl 99 --device HTP0 2>&1 | grep -E "pp512" | sed -E 's/.*pp512 \| +([0-9.]+).*/\1/')
  C=$(timeout 900 ./bin/llama-bench -m models/$M.gguf -p 512 -n 0 -r 3 -ngl 0 --device none -t 4 2>&1 | grep -E "pp512" | sed -E 's/.*pp512 \| +([0-9.]+).*/\1/')
  echo "$M  NPU $N  CPU $C"
done
