cd /data/local/tmp/llama.cpp
export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib
O=/data/local/tmp/ppl.txt; : > $O
run() { echo "== $1" >> $O; shift; timeout 3000 "$@" -f wiki.test.raw -c 512 --chunks 20 -b 512 2>&1 | grep -E "Final estimate|error" >> $O; }
run "unsloth Q4_0 CPU"  ./bin/llama-perplexity -m models/Qwen3.5-4B-Q4_0.gguf -ngl 0 --device none -t 4
run "pure Q4_0 CPU"     ./bin/llama-perplexity -m models/Qwen3.5-4B-Q4_0-pure.gguf -ngl 0 --device none -t 4
GGML_HEXAGON_HMX_Q40_ONLY=1 GGML_HEXAGON_FA_SELECT=1 GGML_HEXAGON_GDN_SELECT=1 run "pure Q4_0 NPU" ./bin/llama-perplexity -m models/Qwen3.5-4B-Q4_0-pure.gguf -ngl 99 --device HTP0
run "BF16 CPU (reference)" ./bin/llama-perplexity -m models/Qwen3.5-4B-BF16.gguf -ngl 0 --device none -t 4
echo DONE >> $O
