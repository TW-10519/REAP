# REAP Pruning Scripts Reference

This document lists all essential scripts kept for REAP pruning on DGX Spark ARM64.

---

## Essential Scripts

### 1. `run_reap_ngc_proper.sh` (3.4K)
**Purpose:** Main pruning execution script using NGC PyTorch container

**Usage:**
```bash
bash run_reap_ngc_proper.sh
```

**What it does:**
- Pulls `nvcr.io/nvidia/pytorch:25.10-py3` container
- Mounts workspace and HuggingFace cache
- Installs REAP dependencies (transformers, datasets, accelerate)
- Runs `reap_prune_arm64.py` inside container
- Outputs pruned model to `/home/nvidia/reap/artifacts/`

**Key Features:**
- GPU verification before and after package installation
- No vLLM installation (pruning-only mode)
- Automatic directory creation

---

### 2. `reap_prune_arm64.py` (4.8K)
**Purpose:** Python entrypoint for pruning with vLLM evaluation disabled

**Usage:**
```bash
python3 reap_prune_arm64.py \
    --model_id huihui-ai/Huihui-gpt-oss-20b-mxfp4-abliterated-v2 \
    --dataset local-prompts \
    --compression_ratio 0.25 \
    --pruning_method reap \
    --samples_per_category 75
```

**Key Modifications:**
- Optional vLLM import (wrapped in try/except)
- `record_pruning_metrics_only=True` to skip vLLM eval
- Supports local-prompts dataset (txt files in `/home/nvidia/reap/prompts/`)

---

### 3. `run_vllm_server.sh` (931 bytes)
**Purpose:** Serve pruned or baseline model via vLLM OpenAI-compatible API

**Usage:**
```bash
# Serve pruned model (default)
bash run_vllm_server.sh

# Edit script to uncomment baseline model line, then:
bash run_vllm_server.sh
```

**Configuration:**
```bash
HOST_PRUNED_MODEL="/home/nvidia/reap/artifacts/.../reap-seed_42-0.25"
# HOST_BASELINE_MODEL="huihui-ai/Huihui-gpt-oss-20b-mxfp4-abliterated-v2"
VLLM_NGC_IMAGE="nvcr.io/nvidia/vllm:26.01-py3"
PORT="8000"
```

**Endpoints:**
- `http://localhost:8000/v1/chat/completions`
- `http://localhost:8000/v1/completions`
- `http://localhost:8000/tokenize`
- `http://localhost:8000/health`

---

### 4. `run_prompt_batch.sh` (605 bytes)
**Purpose:** Run batch prompts against vLLM server and collect statistics

**Usage:**
```bash
# Test pruned model
bash run_prompt_batch.sh /home/nvidia/reap/docs/batch_test_pruned.json

# Test baseline model (after switching server)
bash run_prompt_batch.sh /home/nvidia/reap/docs/batch_test_baseline.json
```

**Environment Variables:**
- `VLLM_SERVER`: Server URL (default: http://localhost:8000)
- `PROMPTS_DIR`: Directory with .txt prompt files (default: /home/nvidia/reap/prompts)
- `MODEL_ID`: Model name for vLLM (must match serving name, e.g., "/models/pruned")
- `OUTPUT_FILE`: JSON output file path

**Output Format:**
```json
{
  "summary": {
    "test_metadata": { "total_prompts": 75, "successful": 75 },
    "aggregate_statistics": { "average_tokens_per_second": 45.2 }
  },
  "results": [
    {
      "prompt_id": 1,
      "prompt": "...",
      "response": "...",
      "statistics": { "tokens_per_second": 47.3, "generation_time_seconds": 2.1 }
    }
  ]
}
```

---

### 5. `docs/run_batch_test.py` (Python script)
**Purpose:** Backend implementation for batch prompt testing

**Key Features:**
- Reads all .txt files from prompts directory
- Extracts numbered prompts from each file
- Uses vLLM tokenizer endpoint for accurate token counting
- Prints full prompt and response to console
- Saves JSON incrementally after each prompt (crash-safe)
- Generates summary statistics

**Sampling Parameters (via env vars):**
```bash
TEMPERATURE=0.8
TOP_P=0.95
MAX_TOKENS=512
REPETITION_PENALTY=1.15
FREQUENCY_PENALTY=0.2
```

---

## Modified Core REAP Files

### 1. `reap/src/reap/observer.py`
**Changes:**
- Added `GptOssMoEObserverHookConfig` for GptOssForCausalLM
- Fixed router logits tuple unpacking
- Added to `OBSERVER_CONFIG_REGISTRY`

### 2. `reap/src/reap/prune.py`
**Changes:**
- Extended fused expert pruning to include bias tensors:
  - `gate_up_proj_bias`
  - `down_proj_bias`
  - `router.bias`
- Auto-add `num_experts`/`n_experts` config aliases for vLLM

### 3. `reap/src/reap/data.py`
**Changes:**
- Made vLLM import optional (wrapped in try/except)

---

## Workflow Summary

```
┌─────────────────────────────────────────────────────────────┐
│ 1. PRUNING (NGC PyTorch Container)                          │
│    run_reap_ngc_proper.sh                                   │
│    └─> reap_prune_arm64.py                                  │
│        └─> observer.py (hook GptOssMLP)                     │
│        └─> prune.py (prune weights + biases)                │
│        └─> Output: artifacts/.../reap-seed_42-0.25/         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 2. SERVING (NGC vLLM Container)                             │
│    run_vllm_server.sh                                       │
│    └─> vllm serve /models/pruned                            │
│        └─> API: http://localhost:8000/v1/chat/completions   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│ 3. TESTING (Host Python)                                    │
│    run_prompt_batch.sh                                      │
│    └─> docs/run_batch_test.py                               │
│        └─> Reads: prompts/*.txt                             │
│        └─> Writes: docs/batch_test_pruned.json              │
└─────────────────────────────────────────────────────────────┘
```

---

## Deleted Obsolete Scripts

The following scripts were removed as they are no longer needed:

- ❌ `patch_vllm_arm64.sh` - Obsolete (NGC vLLM container used instead)
- ❌ `build_pytorch_arm64_cuda.sh` - Obsolete (NGC PyTorch used instead)
- ❌ `build_pytorch_arm64.sh` - Obsolete (NGC PyTorch used instead)
- ❌ `run_reap_gpt_oss_20b_prune.sh` - Obsolete (replaced by run_reap_ngc_proper.sh)
- ❌ `run_reap_gpu_container.sh` - Obsolete (early prototype)
- ❌ `inspect_gptoss_model.py` - Debugging script (no longer needed)

---

## Quick Start

### First Time Setup
```bash
cd /home/nvidia/reap

# Pull NGC containers (one-time)
docker pull nvcr.io/nvidia/pytorch:25.10-py3
docker pull nvcr.io/nvidia/vllm:26.01-py3

# Create prompts directory with .txt files
mkdir -p prompts
echo "1. What is quantum computing?" > prompts/test.txt
```

### Run Pruning
```bash
bash run_reap_ngc_proper.sh
# Output: artifacts/Huihui-gpt-oss-20b-mxfp4-abliterated-v2/local-prompts/pruned_models/reap-seed_42-0.25/
```

### Serve and Test
```bash
# Terminal 1: Start vLLM server
bash run_vllm_server.sh

# Terminal 2: Run batch tests
bash run_prompt_batch.sh docs/batch_test_pruned.json
```

### View Results
```bash
cat docs/batch_test_pruned.json | jq '.summary.aggregate_statistics'
```

---

## Troubleshooting

### Server Returns 404 "Model does not exist"
**Problem:** MODEL_ID mismatch between script and vLLM server

**Solution:** Edit `run_prompt_batch.sh` to set:
```bash
export MODEL_ID="/models/pruned"  # Must match vLLM serving path
```

### Pruning Fails with "CUDA not available"
**Problem:** PyTorch lost GPU support

**Solution:** Use NGC container (don't install packages on host)
```bash
bash run_reap_ngc_proper.sh  # Uses container automatically
```

### vLLM Loading Error "tensor size mismatch"
**Problem:** Old pruned model with unpruned biases

**Solution:** Re-run pruning with updated `prune.py` that includes bias fixes

---

**Last Updated:** February 3, 2026  
**Platform:** NVIDIA DGX Spark (ARM64 + Blackwell GB10)  
**Status:** Production Ready ✅
