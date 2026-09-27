cd /data/local/tmp/llama.cpp
export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib
P="The capital of France is Paris. The capital of Germany is"
GGML_HEXAGON_HMX_Q40_ONLY=1 GGML_HEXAGON_FA_SELECT=1 GGML_HEXAGON_GDN_SELECT=1 timeout 300 ./bin/llama-completion -m models/Qwen3.5-4B-Q4_0.gguf -p "$P" -n 40 --temp 0 -ngl 99 --device HTP0 -c 1024 -no-cnv > /data/local/tmp/out-npu.txt 2>&1
timeout 300 ./bin/llama-completion -m models/Qwen3.5-4B-Q4_0.gguf -p "$P" -n 40 --temp 0 -ngl 0 --device none -c 1024 -t 4 -no-cnv > /data/local/tmp/out-cpu.txt 2>&1
echo done
