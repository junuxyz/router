#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Oracle analysis for GSM8K small/big models.
Excludes both_wrong cases from analysis.
"""

import json
import os
from typing import List, Dict, Any

TRAIN_PATH = "gsm8k_train_small_Qwen_Qwen2.5-1.5B-Instruct_big_Qwen_Qwen2.5-7B-Instruct.jsonl"
TEST_PATH  = "gsm8k_test_small_Qwen_Qwen2.5-1.5B-Instruct_big_Qwen_Qwen2.5-7B-Instruct.jsonl"
SUMMARY_OUT = "oracle_summary.json"


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load jsonl file into list."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def analyze_split(name: str, records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze split, excluding both_wrong cases."""
    total_original = len(records)
    if total_original == 0:
        raise ValueError(f"{name}: empty records")

    filtered_records = []
    n_both_wrong = 0
    
    for r in records:
        s = int(r["small"]["is_correct"])
        b = int(r["big"]["is_correct"])
        if s == 0 and b == 0:
            n_both_wrong += 1
        else:
            filtered_records.append(r)
    
    total = len(filtered_records)
    if total == 0:
        raise ValueError(f"{name}: all records are both_wrong")

    n_only_big_correct = n_only_small_correct = n_both_correct = 0
    small_correct = 0
    big_correct = 0
    oracle_correct = 0

    for r in filtered_records:
        s = int(r["small"]["is_correct"])
        b = int(r["big"]["is_correct"])

        small_correct += s
        big_correct += b

        if s == 0 and b == 1:
            n_only_big_correct += 1
        elif s == 1 and b == 0:
            n_only_small_correct += 1
        elif s == 1 and b == 1:
            n_both_correct += 1
        else:
            raise ValueError(f"Unexpected is_correct pair: s={s}, b={b}")

        if s == 1 or b == 1:
            oracle_correct += 1

    acc_small   = small_correct   / total
    acc_big     = big_correct     / total
    acc_oracle  = oracle_correct  / total

    assert n_only_big_correct + n_only_small_correct + n_both_correct == total, "case count mismatch"
    assert oracle_correct == total, "oracle should be 100% when both_wrong excluded"

    stats = {
        "split": name,
        "total": total,
        "total_original": total_original,
        "counts": {
            "both_wrong": n_both_wrong,
            "only_big_correct": n_only_big_correct,
            "only_small_correct": n_only_small_correct,
            "both_correct": n_both_correct,
        },
        "acc_small": acc_small,
        "acc_big": acc_big,
        "acc_oracle": acc_oracle,
        "num_correct_small": small_correct,
        "num_correct_big": big_correct,
        "num_correct_oracle": oracle_correct,
    }

    print("\n" + "=" * 80)
    print(f"[{name}] Oracle analysis (both_wrong excluded)")
    print("=" * 80)
    print(f"Original samples: {total_original}")
    print(f"Excluded (both_wrong): {n_both_wrong}")
    print(f"Analyzed samples: {total}")
    print(f"small acc   : {acc_small:.2%} ({small_correct}/{total})")
    print(f"big acc     : {acc_big:.2%} ({big_correct}/{total})")
    print(f"oracle acc  : {acc_oracle:.2%} ({oracle_correct}/{total})")
    print("-" * 80)
    print("Case counts (both_wrong excluded):")
    print(f"  both_wrong          : {n_both_wrong} (excluded from analysis)")
    print(f"  only_big_correct    : {n_only_big_correct}")
    print(f"  only_small_correct  : {n_only_small_correct}")
    print(f"  both_correct        : {n_both_correct}")
    print("=" * 80 + "\n")

    return stats


def main():
    for p in [TRAIN_PATH, TEST_PATH]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"File not found: {p}")

    train_records = load_jsonl(TRAIN_PATH)
    test_records  = load_jsonl(TEST_PATH)

    train_stats = analyze_split("train", train_records)
    test_stats  = analyze_split("test",  test_records)

    summary = {
        "train": train_stats,
        "test":  test_stats,
    }

    summary_dir = os.path.dirname(SUMMARY_OUT)
    if summary_dir:
        os.makedirs(summary_dir, exist_ok=True)
    with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Summary saved to: {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
