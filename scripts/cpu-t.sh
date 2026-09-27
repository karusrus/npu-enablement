cd /data/local/tmp/llama.cpp
export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib
for M in Llama-3.2-1B-Instruct-Q4_0 Qwen3.5-4B-Q4_0-pure; do
  timeout 1500 ./bin/llama-bench -m models/$M.gguf -p 512 -n 0 -r 3 -ngl 0 --device none -t 2,4,6,8 2>&1 | grep -E "pp512"
done
echo DONE
