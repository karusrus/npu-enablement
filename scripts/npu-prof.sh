cd /data/local/tmp/llama.cpp
export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib
P="The capital of France is Paris. The capital of Germany is"
GGML_HEXAGON_PROFILE=1 GGML_HEXAGON_HMX_Q40_ONLY=1 GGML_HEXAGON_FA_SELECT=1 GGML_HEXAGON_GDN_SELECT=1 timeout 400 ./bin/llama-completion -m models/Qwen3.5-4B-Q4_0.gguf -p "$P" -n 2 --temp 0 -ngl 99 --device HTP0 -c 1024 -no-cnv -v > /data/local/tmp/out-prof.txt 2>&1
echo done
