#!/bin/bash
set -e

# vLLM server runner (NGC container) for pruned or baseline model.
# Default: pruned model path. Uncomment baseline MODEL_ID to compare.

VLLM_IMAGE="nvcr.io/nvidia/vllm:26.01-py3"
HF_CACHE="/home/nvidia/.cache/huggingface"

# PRUNED MODEL (local path on host)
HOST_PRUNED_MODEL="/home/nvidia/reap/artifacts/Huihui-gpt-oss-20b-mxfp4-abliterated-v2/local-prompts/pruned_models/reap-seed_42-0.25"
# Container path for pruned model
MODEL_ID="/models/pruned"

# BASELINE MODEL (HF Hub) - uncomment to run baseline
# MODEL_ID="huihui-ai/Huihui-gpt-oss-20b-mxfp4-abliterated-v2"

# Optional: serve on a different port
PORT="8000"

# Start vLLM server
exec docker run --rm --gpus all \
  --ipc=host \
  -p ${PORT}:8000 \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -v "${HOST_PRUNED_MODEL}:/models/pruned" \
  "${VLLM_IMAGE}" \
  vllm serve "${MODEL_ID}" \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 8000
