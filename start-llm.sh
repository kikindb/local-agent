#!/bin/bash
# Start the LLM server based on the specified model size (9B or 4B)
# example usage:
# ./start-llm.sh 9B
# ./start-llm.sh 4B
if [ "$1" == "9B" ]; then
  mlx_lm.server \
    --model mlx-community/Qwen3.5-9B-MLX-4bit \
    --port 8080
elif [ "$1" == "4B" ]; then
  mlx_lm.server \
    --model mlx-community/Qwen3.5-4B-MLX-4bit \
    --port 8080
else
  echo "Usage: $0 {9B|4B}"
fi