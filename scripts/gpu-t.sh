cd /data/local/tmp/llama.cpp
export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib
echo "== unsloth CPU t8"; timeout 900 ./bin/llama-bench -m models/Qwen3.5-4B-Q4_0.gguf -p 512 -n 0 -r 3 -ngl 0 --device none -t 8 2>&1 | grep -E "pp512|error"
for M in Llama-3.2-1B-Instruct-Q4_0 Qwen3.5-4B-Q4_0-pure Qwen3.5-4B-Q4_0; do
  echo "== $M GPU"; timeout 900 ./bin/llama-bench -m models/$M.gguf -p 512 -n 0 -r 3 -ngl 99 --device GPUOpenCL 2>&1 | grep -E "pp512|error|rror"
done
echo DONE
