#!/bin/bash
set -e

# Run batch prompts against vLLM server.
# Usage: ./run_prompt_batch.sh <output_json>

OUTPUT_FILE="${1:-/home/nvidia/reap/docs/batch_test_results.json}"

export VLLM_SERVER="http://localhost:8000"
export PROMPTS_DIR="/home/nvidia/reap/prompts"
export OUTPUT_FILE

# If you are serving baseline HF model, set MODEL_ID to the hub id.
# If you are serving pruned local model, set MODEL_ID to match vLLM's model name (container path).
export MODEL_ID="/models/pruned"
# export MODEL_ID="huihui-ai/Huihui-gpt-oss-20b-mxfp4-abliterated-v2"

python3 /home/nvidia/reap/docs/run_batch_test.py
