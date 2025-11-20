#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GSM8K Small/Big inference script (Step 1)

- GSM8K train + test:
  - small model: Qwen 1.5B, zero-shot without CoT 
  - big model  : Qwen 7B, CoT + \boxed{} format

- For each problem:
  - question
  - gsm8k_answer (original answer + "#### answer")
  - gold_answer  (number after "####")
  - For each of small/big:
      - output     (model prediction: short answer for small, CoT for big)
      - pred_answer (extracted number)
      - is_correct  (0/1, string comparison with gold)

- Final results:
  - outputs/gsm8k_train_small_..._big_....jsonl
  - outputs/gsm8k_test_small_..._big_....jsonl
"""

import os
import re
import json
from typing import Dict, Any, List

from vllm import LLM, SamplingParams
from datasets import load_dataset
from tqdm import tqdm


# 0. Fixed settings (modify only here)

# small: Qwen 1.5B (no CoT)
SMALL_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"

# big: Qwen 7B (with CoT)
BIG_MODEL_ID   = "Qwen/Qwen2.5-7B-Instruct"

OUTPUT_DIR = "outputs"

# CoT prompt template for the big model
COT_SUFFIX = """\n\nSolve the following math problem step by step. Put your final answer in \\boxed{{}}.

Problem: {question}

Solution:"""

# max_new_tokens for small / big model
SMALL_MAX_NEW_TOKENS = 512   # for short answer 
BIG_MAX_NEW_TOKENS   = 1024   # for CoT

# Batch size for vLLM inference (higher = better throughput)
BATCH_SIZE = 32


# 1. Answer parsing utilities (same as original)

def normalize_number_str(s: str) -> str:
    """Normalizes number string"""
    if s is None:
        return None
    s = s.replace(",", "").strip()
    # handle '400.' case
    if re.fullmatch(r"-?\d+\.", s):
        s = s[:-1]
    return s


def extract_answer(text: str) -> str:
    """
    Extracts the final number answer from model output.
    example: "The answer is $\\boxed{42}$" -> "42"
    """
    if text is None:
        return None
    # 1) \boxed{42} style
    m = re.search(r"\\boxed{([-+]?\d+\.?\d*)}", text)
    if m:
        return normalize_number_str(m.group(1))

    # 2) #### 42 style
    m = re.search(r"####\s*([-+]?\d+\.?\d*)", text)
    if m:
        return normalize_number_str(m.group(1))

    lower = text.lower()

    # 3) "(the) answer is X" pattern
    m = re.search(r"answer is[:\s]+([-+]?\d+\.?\d*)", lower)
    if m:
        return normalize_number_str(m.group(1))

    # 4) last number
    nums = re.findall(r"[-+]?\d+\.?\d*", text)
    if nums:
        return normalize_number_str(nums[-1])

    return None


def get_gold_answer(answer_text: str) -> str:
    """
    Extracts gold answer from GSM8K answer text
    example: "...\n#### 400" -> "400"
    """
    if answer_text is None:
        return None
    m = re.search(r"#### (.+)", answer_text)
    if not m:
        return None
    gold = m.group(1)
    return normalize_number_str(gold)


# 2. Load vLLM model (replaces load_causal_lm_16bit)

def load_vllm_model(model_id: str, max_model_len: int = None) -> LLM:
    """
    Loads a model using vLLM for high-throughput inference.
    
    vLLM automatically:
    - Optimizes memory with PagedAttention
    - Enables continuous batching
    - Maximizes GPU utilization
    """
    print(f"\nLoading vLLM model: {model_id}")
    
    kwargs = {
        "model": model_id,
        "dtype": "half",  # float16 for efficiency
        "gpu_memory_utilization": 0.90,  # Use 90% of GPU memory
    }
    
    if max_model_len:
        kwargs["max_model_len"] = max_model_len
    
    llm = LLM(**kwargs)
    print(f"   - Model loaded successfully with vLLM")
    
    return llm


# 3. Run inference on a single model for a given split (train/test) - BATCHED VERSION

def run_inference_on_split_vllm(
    split_name: str,
    dataset,      # GSM8K dataset
    llm: LLM,     # vLLM model
    model_type: str,  # "small" or "big"
    save_interval: int = 100,
) -> List[Dict[str, Any]]:
    """
    Runs batched inference on a given split using vLLM.
    
    Key improvements:
    - Batched generation for 10-20x throughput improvement
    - Automatic memory management
    - Continuous batching under the hood
    """
    results = []
    total = len(dataset)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    model_id = SMALL_MODEL_ID if model_type == "small" else BIG_MODEL_ID
    temp_file = os.path.join(
        OUTPUT_DIR,
        f"temp_{split_name}_{model_type}_{model_id.replace('/', '_')}.jsonl"
    )
    
    # Check if file exists (resume from checkpoint)
    processed_indices = set()
    if os.path.exists(temp_file):
        print(f"Found existing checkpoint file: {temp_file}")
        with open(temp_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    results.append(result)
                    processed_indices.add(result["index"])
        print(f"Resuming {model_type} from {len(results)} existing results")

    print(f"\n{'='*80}")
    print(f"Starting vLLM inference for Split='{split_name}', Model='{model_type}' "
          f"(total {total} questions)")
    print(f"{'='*80}")

    # Prepare all prompts and metadata
    prompts = []
    metadata = []
    
    for idx, example in enumerate(dataset):
        if idx in processed_indices:
            continue
            
        question = example["question"]
        gsm8k_answer = example["answer"]
        gold_answer = get_gold_answer(gsm8k_answer)

        # Use different prompt for small/big model
        if model_type == "small":
            prompt = f"Solve this math problem and give only the final numerical answer.\n\nProblem: {question}\n\nAnswer:"
            max_new_tokens = SMALL_MAX_NEW_TOKENS
        else:
            # CoT
            prompt = COT_SUFFIX.format(question=question)
            max_new_tokens = BIG_MAX_NEW_TOKENS
        
        prompts.append(prompt)
        metadata.append({
            "index": idx,
            "question": question,
            "gsm8k_answer": gsm8k_answer,
            "gold_answer": gold_answer,
            "max_new_tokens": max_new_tokens,
        })

    # Set up sampling parameters
    max_tokens = SMALL_MAX_NEW_TOKENS if model_type == "small" else BIG_MAX_NEW_TOKENS
    sampling_params = SamplingParams(
        temperature=0.0,  # greedy decoding (same as do_sample=False)
        max_tokens=max_tokens,
    )

    # Batch inference with vLLM (automatic batching!)
    print(f"Running batched inference on {len(prompts)} prompts...")
    
    with open(temp_file, "a", encoding="utf-8") as f:
        # Process in batches for progress tracking
        for batch_start in tqdm(range(0, len(prompts), BATCH_SIZE), 
                                desc=f"{split_name} - {model_type} generating"):
            batch_end = min(batch_start + BATCH_SIZE, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]
            batch_metadata = metadata[batch_start:batch_end]
            
            # vLLM batched generation - this is where the magic happens!
            outputs = llm.generate(batch_prompts, sampling_params)
            
            # Process outputs
            for output, meta in zip(outputs, batch_metadata):
                full_output = output.outputs[0].text.strip()
                pred_answer = extract_answer(full_output)
                is_correct = int(pred_answer == meta["gold_answer"])

                result = {
                    "index": meta["index"],
                    "question": meta["question"],
                    "gsm8k_answer": meta["gsm8k_answer"],
                    "gold_answer": meta["gold_answer"],
                    "output": full_output,
                    "pred_answer": pred_answer,
                    "is_correct": is_correct,
                }
                
                results.append(result)
                
                # incremental save
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
            
            f.flush()
            
            if (batch_end) % save_interval < BATCH_SIZE:
                num_correct = sum(r["is_correct"] for r in results)
                acc = num_correct / len(results) if results else 0.0
                print(f"\n[Checkpoint][{model_type}] Saved {batch_end}/{len(prompts)} samples - "
                      f"Current accuracy: {acc:.2%} ({num_correct}/{len(results)})")

    num_correct = sum(r["is_correct"] for r in results)
    acc = num_correct / len(results) if results else 0.0
    print(f"\nSplit='{split_name}' results ({model_type} only)")
    print(f"   - Accuracy: {acc:.2%} ({num_correct}/{len(results)})")
    print(f"   - Results saved to: {temp_file}")

    return results


# 4. Merge and save small/big results (same as original)

def merge_and_save(
    split_name: str,
    small_res: List[Dict[str, Any]],
    big_res: List[Dict[str, Any]],
):
    """
    Merges and saves small/big results for the same split into a JSONL file.
    Also removes temporary files after successful merge.
    """
    assert len(small_res) == len(big_res), "small/big results have different lengths"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    out_path = os.path.join(
        OUTPUT_DIR,
        f"gsm8k_{split_name}_small_{SMALL_MODEL_ID.replace('/', '_')}"
        f"_big_{BIG_MODEL_ID.replace('/', '_')}.jsonl",
    )
    print(f"\nSaving merged {split_name} results to: {out_path}")

    num_samples = len(small_res)
    num_correct_small = 0
    num_correct_big = 0

    with open(out_path, "w", encoding="utf-8") as f:
        for s, b in zip(small_res, big_res):
            if s["question"] != b["question"]:
                print("Warning: small/big question mismatch (merge by index)")

            record = {
                "index": s["index"],
                "split": split_name,
                "question": s["question"],
                "gsm8k_answer": s["gsm8k_answer"],
                "gold_answer": s["gold_answer"],
                "small": {
                    "model_id": SMALL_MODEL_ID,
                    "output": s["output"],
                    "pred_answer": s["pred_answer"],
                    "is_correct": s["is_correct"],
                },
                "big": {
                    "model_id": BIG_MODEL_ID,
                    "output": b["output"],
                    "pred_answer": b["pred_answer"],
                    "is_correct": b["is_correct"],
                },
            }
            num_correct_small += int(s["is_correct"])
            num_correct_big += int(b["is_correct"])

            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    acc_small = num_correct_small / num_samples if num_samples > 0 else 0.0
    acc_big = num_correct_big / num_samples if num_samples > 0 else 0.0

    print(f"{split_name} summary")
    print(f"   - Small ({SMALL_MODEL_ID}) Accuracy: {acc_small:.2%} "
          f"({num_correct_small}/{num_samples})")
    print(f"   - Big   ({BIG_MODEL_ID}) Accuracy: {acc_big:.2%} "
          f"({num_correct_big}/{num_samples})")
    
    # Remove temporary files after successful merge
    temp_small = os.path.join(
        OUTPUT_DIR,
        f"temp_{split_name}_small_{SMALL_MODEL_ID.replace('/', '_')}.jsonl"
    )
    temp_big = os.path.join(
        OUTPUT_DIR,
        f"temp_{split_name}_big_{BIG_MODEL_ID.replace('/', '_')}.jsonl"
    )
    
    for temp_file in [temp_small, temp_big]:
        if os.path.exists(temp_file):
            os.remove(temp_file)
            print(f"   - Cleaned up temporary file: {temp_file}")


# 5. main: execution order with vLLM

def main():
    print("\nLoading GSM8K train/test...")
    ds_train = load_dataset("openai/gsm8k", "main", split="train")
    ds_test = load_dataset("openai/gsm8k", "main", split="test")
    print(f"train count: {len(ds_train)}, test count: {len(ds_test)}")

    # 1) small model (1.5B, no CoT) on train
    print("\n" + "="*80)
    print("STAGE 1: Small model on train split")
    print("="*80)
    small_llm = load_vllm_model(SMALL_MODEL_ID)
    small_train_res = run_inference_on_split_vllm(
        "train", ds_train, small_llm, "small", save_interval=100
    )
    del small_llm

    # 2) big model (7B, with CoT) on train
    print("\n" + "="*80)
    print("STAGE 2: Big model on train split")
    print("="*80)
    big_llm = load_vllm_model(BIG_MODEL_ID)
    big_train_res = run_inference_on_split_vllm(
        "train", ds_train, big_llm, "big", save_interval=100
    )
    merge_and_save("train", small_train_res, big_train_res)
    del big_llm

    # 3) small model on test
    print("\n" + "="*80)
    print("STAGE 3: Small model on test split")
    print("="*80)
    small_llm = load_vllm_model(SMALL_MODEL_ID)
    small_test_res = run_inference_on_split_vllm(
        "test", ds_test, small_llm, "small", save_interval=100
    )
    del small_llm

    # 4) big model on test
    print("\n" + "="*80)
    print("STAGE 4: Big model on test split")
    print("="*80)
    big_llm = load_vllm_model(BIG_MODEL_ID)
    big_test_res = run_inference_on_split_vllm(
        "test", ds_test, big_llm, "big", save_interval=100
    )
    del big_llm

    merge_and_save("test", small_test_res, big_test_res)

    print("\nStep 1: GSM8K small/big results generation complete with vLLM!")
    print("="*80)
    print("vLLM benefits achieved:")
    print("  - 10-20x higher throughput via PagedAttention")
    print("  - Automatic continuous batching")
    print("  - Optimized GPU memory utilization")
    print("="*80)


if __name__ == "__main__":
    main()

