cd /data/local/tmp/llama.cpp
export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib
P="The capital of France is Paris. The capital of Germany is"
for M in Llama-3.2-1B-Instruct-Q4_0 Qwen3.5-4B-Q4_0-pure; do
  GGML_HEXAGON_HMX_Q40_ONLY=1 GGML_HEXAGON_FA_SELECT=1 GGML_HEXAGON_GDN_SELECT=1 timeout 300 ./bin/llama-completion -m models/$M.gguf -p "$P" -n 30 --temp 0 -ngl 99 --device HTP0 -c 1024 -no-cnv 2>/dev/null > /data/local/tmp/cmp-npu-$M.txt
  timeout 300 ./bin/llama-completion -m models/$M.gguf -p "$P" -n 30 --temp 0 -ngl 0 --device none -t 4 -c 1024 -no-cnv 2>/dev/null > /data/local/tmp/cmp-cpu-$M.txt
  if cmp -s /data/local/tmp/cmp-npu-$M.txt /data/local/tmp/cmp-cpu-$M.txt; then echo "$M: NPU == CPU"; else echo "$M: DIFFER"; echo "NPU: $(tr '\n' ' ' < /data/local/tmp/cmp-npu-$M.txt | cut -c1-200)"; echo "CPU: $(tr '\n' ' ' < /data/local/tmp/cmp-cpu-$M.txt | cut -c1-200)"; fi
done
