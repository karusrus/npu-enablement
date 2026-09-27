cd /data/local/tmp/llama.cpp
export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib
E="GGML_HEXAGON_HMX_Q40_ONLY=1 GGML_HEXAGON_FA_SELECT=1 GGML_HEXAGON_GDN_SELECT=1"
for M in Llama-3.2-1B-Instruct-Q4_0 Qwen3.5-4B-Q4_0-pure; do
  echo "== $M tg128 NPU"; env $E timeout 900 ./bin/llama-bench -m models/$M.gguf -p 0 -n 128 -r 3 -ngl 99 --device HTP0 2>&1 | grep -E "tg128|rror"
  echo "== $M tg128 GPU"; timeout 900 ./bin/llama-bench -m models/$M.gguf -p 0 -n 128 -r 3 -ngl 99 --device GPUOpenCL 2>&1 | grep -E "tg128|rror"
  echo "== $M tg128 CPU t4,6,8"; timeout 900 ./bin/llama-bench -m models/$M.gguf -p 0 -n 128 -r 3 -ngl 0 --device none -t 4,6,8 2>&1 | grep -E "tg128|rror"
done
M=Llama-3.2-1B-Instruct-Q4_0
echo "== split NPU+GPU pp512 ts 1/1 and 2/1"; env $E timeout 900 ./bin/llama-bench -m models/$M.gguf -p 512 -n 0 -r 3 -ngl 99 -dev HTP0/GPUOpenCL -ts 1/1,2/1 2>&1 | grep -E "pp512|rror"
echo DONE
