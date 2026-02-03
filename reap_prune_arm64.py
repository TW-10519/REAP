#!/usr/bin/env python3
"""
Minimal REAP pruning launcher for ARM64 unified memory systems.
Avoids vllm CUDA import issues by only running observation + pruning (no eval).
"""
import os
import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Add REAP src to path
reap_root = Path(__file__).parent / "reap"
sys.path.insert(0, str(reap_root / "src"))

# Set HF cache
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/.cache/huggingface/transformers")

# Import only what we need (avoid eval module which imports vllm)
from transformers import AutoTokenizer, AutoModelForCausalLM, HfArgumentParser
from accelerate.utils import set_seed
import torch

from reap.args import (
    ReapArgs, ModelArgs, DatasetArgs, ObserverArgs, 
    ClusterArgs, EvalArgs, PruneArgs,
)
from reap.main import record_activations, create_results_directory
from reap.prune import prune, get_pruned_model_dir, dump_args_to_yaml
from reap.data import DATASET_REGISTRY, build_local_prompts_dataset
from reap.model_util import patched_model_map, MODEL_ATTRS
import dataclasses

logger.info("=" * 80)
logger.info("REAP Pruning - ARM64 Unified Memory Mode (vLLM Eval Disabled)")
logger.info("=" * 80)

def main():
    # Parse arguments
    parser = HfArgumentParser(
        (
            ReapArgs,
            ModelArgs,
            DatasetArgs,
            ObserverArgs,
            ClusterArgs,
            EvalArgs,
            PruneArgs,
        )
    )
    (reap_args, model_args, ds_args, obs_args, cluster_args, 
     eval_args, prune_args) = parser.parse_args_into_dataclasses()
    
    # Override eval to false
    reap_args.do_eval = False
    reap_args.smoke_test = False
    
    logger.info(f"Model: {model_args.model_name}")
    logger.info(f"Dataset: {ds_args.dataset_name}")
    logger.info(f"Compression Ratio: {cluster_args.compression_ratio}")
    logger.info(f"Pruning Method: {prune_args.prune_method}")
    logger.info("")
    
    set_seed(reap_args.seed)
    results_dir = create_results_directory(model_args.model_name, ds_args.dataset_name)
    
    # Load model (try local first, fall back to download if needed)
    model_name = patched_model_map(model_args.model_name)
    logger.info(f"[1/4] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
            local_files_only=True,
        )
    except Exception as e:
        logger.info(f"Local cache incomplete, downloading missing files...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
            local_files_only=False,
        )
    logger.info(f"✓ Model loaded: {model.__class__.__name__}")
    
    # Record activations (observer)
    logger.info(f"[2/4] Recording expert activations from {ds_args.dataset_name}...")
    observer_data = record_activations(
        model, tokenizer, reap_args, model_args, ds_args, obs_args, results_dir
    )
    logger.info(f"✓ Activations recorded for {len(observer_data)} layers")
    
    # Pruning
    logger.info(f"[3/4] Pruning experts with {prune_args.prune_method}...")
    total_experts = len(observer_data[next(iter(observer_data))]["expert_frequency"])
    n_experts_to_prune = prune_args.n_experts_to_prune
    if n_experts_to_prune is None:
        n_experts_to_prune = int(total_experts * cluster_args.compression_ratio)
        logger.info(f"Calculated experts to prune: {n_experts_to_prune}/{total_experts}")
    
    pruned_model_dir = get_pruned_model_dir(
        results_dir, n_experts_to_prune, total_experts, 
        prune_args, reap_args.seed, obs_args.renormalize_router_weights
    )
    
    if (pruned_model_dir.exists() and 
        list(pruned_model_dir.glob("*.safetensors")) and 
        not prune_args.overwrite_pruned_model):
        logger.info(f"✓ Pruned model already exists at: {pruned_model_dir}")
    else:
        prune(
            observer_data, model, tokenizer, reap_args, prune_args,
            n_experts_to_prune, pruned_model_dir,
        )
        tokenizer.save_pretrained(pruned_model_dir)
        dump_args_to_yaml(
            pruned_model_dir, reap_args, ds_args, obs_args, model_args, 
            eval_args, prune_args, cluster_args,
        )
        logger.info(f"✓ Pruned model saved to: {pruned_model_dir}")
    
    logger.info("")
    logger.info("[4/4] Pruning complete!")
    logger.info(f"Output: {pruned_model_dir}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main()
