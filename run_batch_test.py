#!/usr/bin/env python3
"""
Prompt Batch Testing Script
Extracts prompts from txt files in the prompts folder and runs them against vLLM server
Records both questions and answers with full statistics
"""

import json
import os
import re
import time
from pathlib import Path
from typing import List, Dict
import requests

# Configuration (override via env vars)
VLLM_SERVER = os.getenv("VLLM_SERVER", "http://localhost:8000")
PROMPTS_DIR = os.getenv("PROMPTS_DIR", "./prompts")
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "./batch_test_results.json")
MODEL_ID = os.getenv(
    "MODEL_ID", "huihui-ai/Huihui-gpt-oss-20b-mxfp4-abliterated-v2"
)

# Sampling parameters (override via env vars)
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.8"))
TOP_P = float(os.getenv("TOP_P", "0.95"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.15"))
FREQUENCY_PENALTY = float(os.getenv("FREQUENCY_PENALTY", "0.2"))


def extract_prompts_from_file(file_path: Path) -> List[Dict[str, str]]:
    """
    Extract individual prompts from a txt file.
    Format: numbered prompts separated by blank lines
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    prompts = []
    
    # Split by numbered items (1. 2. 3. etc.) or blank lines
    # Match patterns like "1." or "1)" at the start of a line
    sections = re.split(r'\n(?=\d+[\.\)])', content)
    
    for section in sections:
        section = section.strip()
        if not section or len(section) < 10:  # Skip empty or very short sections
            continue
        
        # Remove the leading number
        prompt_text = re.sub(r'^\d+[\.\)]\s*', '', section)
        prompt_text = prompt_text.strip()
        
        if prompt_text and not prompt_text.startswith('LEVEL'):  # Skip header lines
            prompts.append({
                "text": prompt_text,
                "source_file": file_path.name
            })
    
    return prompts


def load_all_prompts(prompts_dir: str) -> List[Dict[str, str]]:
    """Load all prompts from all txt files in the directory."""
    prompts_path = Path(prompts_dir)
    all_prompts = []
    
    for txt_file in sorted(prompts_path.glob("*.txt")):
        file_prompts = extract_prompts_from_file(txt_file)
        all_prompts.extend(file_prompts)
        print(f"Loaded {len(file_prompts)} prompts from {txt_file.name}")
    
    return all_prompts


def wait_for_server(server_url: str, timeout: int = 30):
    """Wait for vLLM server to be ready."""
    print(f"Checking vLLM server at {server_url}...")
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"{server_url}/health", timeout=5)
            if response.status_code == 200:
                print("✓ Server is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    
    raise TimeoutError(f"Server did not become ready within {timeout} seconds")


def tokenize_text(text: str) -> int:
    """Get token count using vLLM's tokenizer endpoint for consistency."""
    url = f"{VLLM_SERVER}/tokenize"
    payload = {"prompt": text}
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        return len(result.get("tokens", []))
    except Exception as e:
        print(f"Warning: Could not tokenize with vLLM endpoint: {e}")
        # Fallback: approximate token count
        return len(text.split())


def generate_text(prompt: str) -> Dict:
    """Generate text using vLLM OpenAI-compatible API."""
    url = f"{VLLM_SERVER}/v1/chat/completions"
    
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "repetition_penalty": REPETITION_PENALTY,
        "frequency_penalty": FREQUENCY_PENALTY,
        "stream": False
    }
    
    # Get prompt token count using vLLM tokenizer
    prompt_tokens = tokenize_text(prompt)
    
    start_time = time.time()
    response = requests.post(url, json=payload, timeout=300)
    end_time = time.time()
    
    response.raise_for_status()
    result = response.json()
    
    # Extract statistics
    choice = result["choices"][0]
    generated_text = choice["message"]["content"]
    
    # Get response token count using vLLM tokenizer
    response_tokens = tokenize_text(generated_text)
    
    return {
        "generated_text": generated_text,
        "finish_reason": choice["finish_reason"],
        "prompt_tokens": prompt_tokens,
        "completion_tokens": response_tokens,
        "total_tokens": prompt_tokens + response_tokens,
        "generation_time": end_time - start_time,
        "tokens_per_second": response_tokens / (end_time - start_time) if (end_time - start_time) > 0 else 0
    }


def save_results_incrementally(results: List[Dict], total_tokens: int, total_time: float, total_prompts: int):
    """Save results to JSON file incrementally after each prompt."""
    successful_results = [r for r in results if 'error' not in r]
    summary = {
        "test_metadata": {
            "total_prompts": total_prompts,
            "successful": len(successful_results),
            "failed": total_prompts - len(successful_results),
            "sampling_params": {
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "max_tokens": MAX_TOKENS,
                "repetition_penalty": REPETITION_PENALTY,
                "frequency_penalty": FREQUENCY_PENALTY
            }
        },
        "aggregate_statistics": {
            "total_completion_tokens": total_tokens,
            "total_generation_time_seconds": round(total_time, 2),
            "average_tokens_per_second": round(total_tokens / total_time, 2) if total_time > 0 else 0,
            "average_tokens_per_prompt": round(total_tokens / len(successful_results), 2) if successful_results else 0,
            "average_time_per_prompt": round(total_time / len(successful_results), 2) if successful_results else 0
        }
    }
    
    output_data = {
        "summary": summary,
        "results": results
    }
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)


def main():
    print("=" * 70)
    print("Prompt Batch Testing Script")
    print("=" * 70)
    print(f"Server: {VLLM_SERVER}")
    print(f"Prompts directory: {PROMPTS_DIR}")
    print(f"Output file: {OUTPUT_FILE}")
    print("=" * 70)
    print()
    
    # Wait for server to be ready
    try:
        wait_for_server(VLLM_SERVER)
    except TimeoutError as e:
        print(f"Error: {e}")
        print("Make sure the vLLM server is running: ./run_ngc.sh")
        return
    
    # Load all prompts
    print("\nLoading prompts...")
    prompts = load_all_prompts(PROMPTS_DIR)
    print(f"\nTotal prompts loaded: {len(prompts)}")
    print("=" * 70)
    
    # Process each prompt
    results = []
    total_tokens = 0
    total_time = 0
    
    for i, prompt_data in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Processing prompt from {prompt_data['source_file']}")
        print(f"Prompt: {prompt_data['text']}")
        print("-" * 70)
        
        try:
            stats = generate_text(prompt_data['text'])
            
            result = {
                "prompt_id": i,
                "source_file": prompt_data["source_file"],
                "prompt": prompt_data["text"],
                "response": stats["generated_text"],
                "statistics": {
                    "prompt_tokens": stats["prompt_tokens"],
                    "completion_tokens": stats["completion_tokens"],
                    "total_tokens": stats["total_tokens"],
                    "generation_time_seconds": round(stats["generation_time"], 3),
                    "tokens_per_second": round(stats["tokens_per_second"], 2),
                    "finish_reason": stats["finish_reason"]
                }
            }
            results.append(result)
            
            total_tokens += stats['completion_tokens']
            total_time += stats['generation_time']
            
            # Print the response
            print(f"Response: {stats['generated_text']}")
            print("-" * 70)
            print(f"✓ Generated {stats['completion_tokens']} tokens in {stats['generation_time']:.2f}s")
            print(f"✓ Speed: {stats['tokens_per_second']:.2f} tok/s")
            
        except Exception as e:
            print(f"✗ Error: {str(e)}")
            results.append({
                "prompt_id": i,
                "source_file": prompt_data["source_file"],
                "prompt": prompt_data["text"],
                "error": str(e)
            })
        
        # Save results incrementally after each prompt
        save_results_incrementally(results, total_tokens, total_time, len(prompts))
    
    # Print summary
    successful_results = [r for r in results if 'error' not in r]
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total prompts: {len(prompts)}")
    print(f"Successful: {len(successful_results)}")
    print(f"Failed: {len(prompts) - len(successful_results)}")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Total time: {round(total_time, 2)}s")
    print(f"Average speed: {round(total_tokens / total_time, 2) if total_time > 0 else 0} tok/s")
    print(f"Average tokens per prompt: {round(total_tokens / len(successful_results), 2) if successful_results else 0}")
    print(f"\nResults saved to: {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
