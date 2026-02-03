#!/bin/bash
set -e

# REAP Pruning on DGX Spark using NGC PyTorch Container (25.10+ with CUDA 13 support)
# This script follows NVIDIA's recommended NGC + REAP workflow

REAP_ROOT="/home/nvidia/reap"
# Use newer NGC container with better Blackwell support
NGC_CONTAINER="nvcr.io/nvidia/pytorch:25.10-py3"

# Model and pruning config
MODEL_ID="huihui-ai/Huihui-gpt-oss-20b-mxfp4-abliterated-v2"
COMPRESSION_RATIO="0.25"
SAMPLES_PER_CATEGORY="75"
SEED="42"

echo "=============================================================================="
echo "REAP Pruning on DGX Spark (Grace Blackwell) using NGC PyTorch Container"
echo "=============================================================================="
echo "Container: $NGC_CONTAINER"
echo "Model: $MODEL_ID"
echo "Compression Ratio: $COMPRESSION_RATIO"
echo "GPU: Grace Blackwell (sm_90)"
echo ""

# Pull the latest NGC container if not present
echo "[0/4] Ensuring NGC container is available..."
docker pull "$NGC_CONTAINER" 2>&1 | tail -3

# Run REAP inside NGC container with proper GPU support
echo "[1/4] Starting NGC container with GPU support..."
docker run --rm \
  --gpus all \
  --ipc=host \
  --shm-size=8g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$REAP_ROOT:/workspace/reap" \
  -v "/home/nvidia/.cache/huggingface:/root/.cache/huggingface" \
  -w /workspace/reap \
  "$NGC_CONTAINER" \
  bash -c '
    set -e
    
    echo "[1/4] Verifying NGC PyTorch + GPU..."
    python3 << EOF
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name()}")
print(f"Supported archs: {torch.cuda.get_arch_list()}")
assert torch.cuda.is_available(), "GPU not available!"
EOF
    
    echo "[2/4] Installing REAP dependencies (skipping vllm for pruning-only)..."
    cd /workspace/reap/reap
    
    # Install only what we need for pruning (no vllm, no eval dependencies)
    pip install -q \
      "transformers==4.55.0" \
      "datasets>=3.6.0,<4.0.0" \
      "accelerate>=1.7.0" \
      "matplotlib>=3.10.3" \
      "python-dotenv>=1.1.0" \
      "seaborn>=0.13.2" \
      "umap-learn>=0.5.7" \
      "wandb>=0.21.1" \
      "tqdm" \
      "safetensors" \
      "pyyaml" 2>&1 | tail -10
    
    echo "[3/4] Verifying GPU still available after install..."
    python3 -c "import torch; assert torch.cuda.is_available(); print(f\"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()}\")"
    
    echo "[4/4] Running REAP pruning with GPU acceleration..."
    export PYTHONPATH=/workspace/reap/reap/src:\$PYTHONPATH
    export HF_HOME=/root/.cache/huggingface
    
    cd /workspace/reap
    python3 /workspace/reap/reap_prune_arm64.py \
      --model-name huihui-ai/Huihui-gpt-oss-20b-mxfp4-abliterated-v2 \
      --dataset-name local-prompts \
      --local_prompts_dir /workspace/reap/prompts \
      --compression-ratio 0.25 \
      --prune-method reap \
      --samples_per_category 75 \
      --record_pruning_metrics_only true \
      --seed 42 \
      --profile false
    
    echo ""
    echo "✅ REAP pruning completed successfully!"
  '

echo ""
echo "=============================================================================="
echo "Pruning complete! Artifacts saved to:"
echo "$REAP_ROOT/reap/artifacts/Huihui-gpt-oss-20b-mxfp4-abliterated-v2/local-prompts/pruned_models/"
echo "=============================================================================="
