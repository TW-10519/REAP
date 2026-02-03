# REAP Pruning on DGX Spark - Final Package

This directory contains all essential files for REAP MoE pruning on NVIDIA DGX Spark (ARM64 + Blackwell GB10).

---

## Directory Structure

```
reap_final/
├── README.md                              # This file
├── REAP_PRUNING_REPORT_DGX_SPARK.md      # Complete technical report (26KB)
├── SCRIPTS_REFERENCE.md                   # Scripts usage guide (8.4KB)
│
├── run_reap_ngc_proper.sh                # Main pruning script (NGC PyTorch)
├── reap_prune_arm64.py                    # Python pruning entrypoint
├── run_vllm_server.sh                     # Model serving script (NGC vLLM)
├── run_prompt_batch.sh                    # Batch testing wrapper
├── run_batch_test.py                      # Batch testing implementation
│
└── modified_reap_files/                   # Modified REAP source files
    ├── observer.py                        # Added GptOss observer config
    ├── prune.py                           # Fixed bias tensor pruning
    └── data.py                            # Made vLLM import optional
```

---

## Documentation

### 1. REAP_PRUNING_REPORT_DGX_SPARK.md
**Comprehensive 749-line technical report covering:**
- Hardware environment and challenges
- Model details (13GB MXFP4 → 31GB BF16)
- All 5 workarounds for ARM64 limitations
- NGC container strategy
- Complete execution flow with code
- All code modifications documented
- Native observer support (Qwen3, Mixtral, etc.)
- Results and performance analysis
- Lessons learned and future improvements

### 2. SCRIPTS_REFERENCE.md
**Practical 270-line usage guide covering:**
- Detailed explanation of each script
- Usage examples and environment variables
- Workflow diagrams
- Quick start guide
- Troubleshooting section

---

## Quick Start

### Prerequisites
```bash
# Ensure NGC containers are available
docker pull nvcr.io/nvidia/pytorch:25.10-py3
docker pull nvcr.io/nvidia/vllm:26.01-py3

# Create prompts directory (if not exists)
mkdir -p /home/nvidia/reap/prompts
```

### Step 1: Run Pruning
```bash
cd /home/nvidia/reap
bash reap_final/run_reap_ngc_proper.sh
```

Output: `/home/nvidia/reap/artifacts/.../reap-seed_42-0.25/`

### Step 2: Serve Pruned Model
```bash
bash reap_final/run_vllm_server.sh
```

### Step 3: Test with Batch Prompts
```bash
bash reap_final/run_prompt_batch.sh /home/nvidia/reap/docs/batch_test_pruned.json
```

---

## Essential Scripts

### 1. `run_reap_ngc_proper.sh`
Main pruning script using NGC PyTorch container.

**What it does:**
- Pulls NGC PyTorch 25.10 container
- Installs REAP dependencies (transformers, datasets, accelerate)
- Runs pruning on Huihui-gpt-oss-20b-mxfp4-abliterated-v2
- Outputs pruned model to artifacts directory

**Configuration:**
```bash
MODEL_ID="huihui-ai/Huihui-gpt-oss-20b-mxfp4-abliterated-v2"
COMPRESSION_RATIO="0.25"  # 25% of experts removed (32→24)
SAMPLES_PER_CATEGORY="75"
```

### 2. `reap_prune_arm64.py`
Python entrypoint for pruning with vLLM evaluation disabled.

**Key features:**
- Optional vLLM import (ARM64 compatible)
- `record_pruning_metrics_only=True`
- Supports local-prompts dataset

### 3. `run_vllm_server.sh`
Serves pruned model via vLLM OpenAI-compatible API.

**Endpoints:**
- `http://localhost:8000/v1/chat/completions`
- `http://localhost:8000/v1/completions`
- `http://localhost:8000/tokenize`

### 4. `run_prompt_batch.sh` + `run_batch_test.py`
Batch inference testing against vLLM server.

**Features:**
- Prints full prompt and response
- Incremental JSON saving (crash-safe)
- Accurate token counting via vLLM tokenizer

---

## Modified REAP Files

### modified_reap_files/observer.py
**Changes:**
- Added `GptOssMoEObserverHookConfig` for GptOssForCausalLM
- Fixed router logits tuple unpacking
- Registered in `OBSERVER_CONFIG_REGISTRY`

**To apply:** Copy to `/path/to/reap/reap/src/reap/observer.py`

### modified_reap_files/prune.py
**Changes:**
- Extended fused expert pruning to include bias tensors:
  - `gate_up_proj_bias`
  - `down_proj_bias`
  - `router.bias`
- Auto-add `num_experts`/`n_experts` config aliases

**To apply:** Copy to `/path/to/reap/reap/src/reap/prune.py`

### modified_reap_files/data.py
**Changes:**
- Made vLLM import optional (wrapped in try/except)

**To apply:** Copy to `/path/to/reap/reap/src/reap/data.py`

---

## Key Results

### Model Size Comparison
| Model | Size | Experts | Quantization |
|-------|------|---------|--------------|
| Original MXFP4 | 13 GB | 32 | 4-bit |
| Pruned BF16 | 31 GB | 24 | 16-bit |
| Expected MXFP4 (pruned) | ~9.75 GB | 24 | 4-bit |

**Note:** Size increased due to MXFP4→BF16 dequantization (Triton 3.4.0+ required for MXFP4).

### Pruning Statistics
- **Compression Ratio:** 0.25 (25% of experts removed)
- **Experts per Layer:** 32 → 24
- **Total Layers:** 24 MoE layers
- **Samples Used:** 75 prompts
- **Pruning Time:** ~5 minutes

### Resource Usage
- **GPU Memory:** 30.1 GiB during serving
- **KV Cache:** 1,598,816 tokens available
- **Max Concurrency:** 23.99x at 131K tokens/request

---

## Key Challenges & Solutions

### Challenge 1: vLLM Cannot Be Installed on ARM64
**Solution:** Skipped vLLM during pruning, used separate NGC vLLM container for serving.

### Challenge 2: PyTorch GPU Version Conflicts
**Solution:** Used NGC PyTorch container with pre-built CUDA support.

### Challenge 3: MXFP4 Dequantization
**Root Cause:** NGC PyTorch 25.10 has Triton 3.1.0 (MXFP4 requires ≥3.4.0)  
**Impact:** Model increased from 13GB → 31GB  
**Future Fix:** Build custom NGC container with Triton 3.4.0+ or apply post-pruning quantization

### Challenge 4: GptOss Observer Not Registered
**Solution:** Added custom observer configuration after model introspection.

### Challenge 5: Fused Expert Bias Tensors Not Pruned
**Solution:** Extended pruning logic to handle bias tensors for fused experts.

---

## Native Observer Support

REAP includes **out-of-the-box support** for these MoE architectures (no code changes needed):

| Model Family | Supported | Fused Experts |
|--------------|-----------|---------------|
| **Qwen3-MoE** | ✅ Yes | ✅ |
| **GptOss (Huihui)** | ✅ Yes | ✅ |
| **Llama4-MoE** | ✅ Yes | ✅ |
| **Mixtral** | ✅ Yes | ❌ |
| **DeepSeek-V2** | ✅ Yes | ❌ |
| **Ernie4.5-MoE** | ✅ Yes | ❌ |
| **GLM4-MoE** | ✅ Yes | ❌ |

---

## Usage Notes

### Running from This Directory
All scripts expect to be run from `/home/nvidia/reap`, not from `reap_final/`:

```bash
# Correct
cd /home/nvidia/reap
bash reap_final/run_reap_ngc_proper.sh

# Incorrect (will fail)
cd /home/nvidia/reap/reap_final
bash run_reap_ngc_proper.sh
```

### Applying Modified Files
To use the modified REAP source files:

```bash
# Backup originals
cp /home/nvidia/reap/reap/src/reap/observer.py /home/nvidia/reap/reap/src/reap/observer.py.bak
cp /home/nvidia/reap/reap/src/reap/prune.py /home/nvidia/reap/reap/src/reap/prune.py.bak
cp /home/nvidia/reap/reap/src/reap/data.py /home/nvidia/reap/reap/src/reap/data.py.bak

# Apply modified files
cp reap_final/modified_reap_files/observer.py /home/nvidia/reap/reap/src/reap/
cp reap_final/modified_reap_files/prune.py /home/nvidia/reap/reap/src/reap/
cp reap_final/modified_reap_files/data.py /home/nvidia/reap/reap/src/reap/
```

---

## Troubleshooting

### Error: "Model does not exist"
**Cause:** MODEL_ID mismatch  
**Fix:** Edit `run_prompt_batch.sh` to set `MODEL_ID="/models/pruned"`

### Error: "CUDA not available"
**Cause:** PyTorch lost GPU support  
**Fix:** Use NGC container (already handled in scripts)

### Error: "tensor size mismatch"
**Cause:** Old pruned model with unpruned biases  
**Fix:** Delete old model and re-run with updated `prune.py`

---

## Support

For questions or issues:
- Review `REAP_PRUNING_REPORT_DGX_SPARK.md` for detailed technical information
- Check `SCRIPTS_REFERENCE.md` for usage examples
- Consult troubleshooting sections in both documents

---

## License

REAP framework: Cerebras Research  
This package: Documentation and scripts for DGX Spark deployment

---

**Package Created:** February 3, 2026  
**Platform:** NVIDIA DGX Spark (ARM64 + Blackwell GB10)  
