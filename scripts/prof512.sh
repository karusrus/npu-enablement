cd /data/local/tmp/llama.cpp
export LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib
P=$(yes "The quick brown fox jumps over the lazy dog near the river bank." | head -40 | tr '\n' ' ')
GGML_HEXAGON_PROFILE=1 GGML_HEXAGON_HMX_Q40_ONLY=1 GGML_HEXAGON_FA_SELECT=1 GGML_HEXAGON_GDN_SELECT=1 timeout 600 ./bin/llama-completion -m models/Qwen3.5-4B-Q4_0.gguf -p "$P" -n 1 --temp 0 -ngl 99 --device HTP0 -c 1024 -b 512 -ub 512 -no-cnv -v > /data/local/tmp/out-prof512.txt 2>&1
echo done
