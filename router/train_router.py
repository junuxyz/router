import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from datetime import datetime

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import wandb

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_environment():
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available")
    
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    
    if "A100" in gpu_name:
        batch_size = 32
        gradient_accumulation = 1
    else:
        batch_size = 16
        gradient_accumulation = 2
    
    return batch_size, gradient_accumulation


def setup_wandb(project_name=None):
    if project_name is None:
        project_name = f"train_router-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    try:
        wandb.login()
    except:
        print("WandB login failed, continuing without tracking")
        return None
    
    run = wandb.init(
        project=project_name,
        name=f"cascade-router-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            "dataset": "GSM8K",
            "small_model": "Qwen2.5-1.5B",
            "big_model": "Qwen2.5-7B",
            "task": "cascade_routing",
        }
    )
    
    print(f"WandB: {run.url}")
    return run


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    # Read from outputs/ folder
    import os
    file_path = os.path.join("outputs", path)
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_oracle_summary(json_path="oracle_summary.json"):
    """Load pre-computed oracle statistics"""
    import os
    file_path = os.path.join("outputs", json_path)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Warning: {file_path} not found, will compute statistics from data")
        return None


def prepare_data(jsonl_path: str, split_name: str, question_only: bool = False) -> Tuple[List[Dict], List[Dict], List[str], List[int]]:
    print(f"Loading {split_name} data...")
    all_records = load_jsonl(jsonl_path)
    
    filtered_records = []
    texts = []
    labels = []
    
    stats = {
        "total": len(all_records),
        "only_big_correct": 0,
        "only_small_correct": 0,
        "both_correct": 0,
        "both_wrong": 0,
    }
    
    for r in all_records:
        small_ok = int(r["small"]["is_correct"])
        big_ok = int(r["big"]["is_correct"])
        
        # Filter cases where both models are wrong (to match oracle_summary.json)
        if small_ok == 0 and big_ok == 0:
            stats["both_wrong"] += 1
            continue
        
        if small_ok == 0 and big_ok == 1:
            stats["only_big_correct"] += 1
        elif small_ok == 1 and big_ok == 0:
            stats["only_small_correct"] += 1
        else:  # small_ok == 1 and big_ok == 1
            stats["both_correct"] += 1
        
        if question_only:
            text = f"Question: {r['question']}"
        else:
            text = f"Question: {r['question']}\n\nSmall Model Answer:\n{r['small']['output']}"
        need_big = 1 if (small_ok == 0 and big_ok == 1) else 0
        
        filtered_records.append(r)
        texts.append(text)
        labels.append(need_big)
    
    stats["need_big_1"] = sum(labels)
    stats["need_big_0"] = len(labels) - sum(labels)
    
    if wandb.run:
        for k, v in stats.items():
            wandb.log({f"{split_name}/{k}": v})
    
    print(f"  Total: {stats['total']}")
    print(f"  Labels: need_big=1: {stats['need_big_1']}, need_big=0: {stats['need_big_0']}")
    
    return all_records, filtered_records, texts, labels


class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        # Each item is a dictionary with tensor-ized input components and the label
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)


def train_and_predict(
    model_name: str,
    train_texts: List[str],
    train_labels: List[int],
    test_texts: List[str],
    batch_size: int,
    gradient_accumulation: int,
) -> Tuple[np.ndarray, Any]:
    print(f"\nTraining: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params/1e6:.1f}M")
    
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors=None
    )
    test_encodings = tokenizer(
        test_texts,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors=None
    )
    
    train_dataset = TokenizedDataset(train_encodings, train_labels)
    test_dataset = TokenizedDataset(test_encodings, [0] * len(test_texts))
    
    output_dir = f"./outputs/{model_name.replace('/', '_')}"
    
    # Get hyperparameters from wandb.config if available, else use defaults
    num_epochs = wandb.config.get("num_epochs", 3) if wandb.run else 3
    learning_rate = wandb.config.get("learning_rate", 2e-5) if wandb.run else 2e-5
    warmup_steps = wandb.config.get("warmup_steps", 100) if wandb.run else 100
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="no",
        report_to="wandb" if wandb.run else "none",
        run_name=f"{model_name.split('/')[-1]}",
        fp16=False,
        dataloader_num_workers=2,
        disable_tqdm=False,
    )
    
    print(f"  Batch size: {batch_size} × {gradient_accumulation} = {batch_size * gradient_accumulation}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Epochs: {num_epochs}, Warmup steps: {warmup_steps}")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )
    
    train_result = trainer.train()
    print(f"  Training complete, loss: {train_result.training_loss:.4f}")
    
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=-1)
    logits = predictions.predictions
    if np.isnan(logits).any():
        print("Warning: NaN detected in predictions. Replacing with zeros.")
        logits = np.nan_to_num(logits, nan=0.0)
    pred_probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    
    print(f"  Predictions: 0={sum(pred_labels==0)}, 1={sum(pred_labels==1)}")
    
    return pred_labels, {
        "train_loss": train_result.training_loss,
        "num_params": num_params,
        "pred_probs": pred_probs.tolist(),
    }


def evaluate_cascade(
    all_records: List[Dict[str, Any]],
    filtered_records: List[Dict[str, Any]],
    predictions: np.ndarray,
    model_name: str,
    oracle_summary: Dict[str, Any] = None,
    split_name: str = "test",
) -> Dict[str, float]:
    cascade_correct = 0
    big_calls = 0
    
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    for record, pred in zip(filtered_records, predictions):
        small_ok = int(record["small"]["is_correct"])
        big_ok = int(record["big"]["is_correct"])
        needs_big = small_ok == 0 and big_ok == 1
        decision = int(pred)
        
        if decision == 0:
            cascade_correct += small_ok
        else:
            cascade_correct += big_ok
            big_calls += 1
        
        if needs_big:
            if decision == 1:
                true_positives += 1
            else:
                false_negatives += 1
        else:
            if decision == 1:
                false_positives += 1
            else:
                true_negatives += 1
    
    if oracle_summary and split_name in oracle_summary:
        summary = oracle_summary[split_name]
        oracle_correct = summary["num_correct_oracle"]
        big_correct = summary["num_correct_big"]
        small_correct = summary["num_correct_small"]
        oracle_big_calls = summary["counts"]["only_big_correct"]
        total_all = len(filtered_records)
    else:
        oracle_correct = 0
        big_correct = 0
        small_correct = 0
        oracle_big_calls = 0
        
        for record in filtered_records:
            small_ok = int(record["small"]["is_correct"])
            big_ok = int(record["big"]["is_correct"])
            
            oracle_correct += max(small_ok, big_ok)
            if small_ok == 0 and big_ok == 1:
                oracle_big_calls += 1
            
            big_correct += big_ok
            small_correct += small_ok
        
        total_all = len(filtered_records)
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    results = {
        "cascade_acc": cascade_correct / total_all,
        "oracle_acc": oracle_correct / total_all,
        "always_big_acc": big_correct / total_all,
        "always_small_acc": small_correct / total_all,
        "big_call_rate": big_calls / total_all,
        "oracle_big_rate": oracle_big_calls / total_all,
        "accuracy_gap_from_oracle": (oracle_correct - cascade_correct) / total_all,
        "accuracy_gain_vs_small": (cascade_correct - small_correct) / total_all,
        "accuracy_gap_vs_big": (big_correct - cascade_correct) / total_all,
        "cost_saving_vs_always_big": 1.0 - (big_calls / total_all),
        "cost_overhead_vs_oracle": (big_calls - oracle_big_calls) / total_all,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "threshold": 0.5,
    }
    
    if wandb.run:
        wandb.log({f"{model_name}/{k}": v for k, v in results.items()})
    
    return results


def print_results(model_name: str, results: Dict[str, float], model_size: str, extra_info: Dict):
    print(f"\n{model_name} ({model_size})")
    print(f"  Cascade accuracy: {results['cascade_acc']*100:.2f}%")
    print(f"  Oracle (upper bound): {results['oracle_acc']*100:.2f}%")
    print(f"  Gap from Oracle: {results['accuracy_gap_from_oracle']*100:.2f}%")
    print(f"  Big call rate: {results['big_call_rate']*100:.2f}%")
    print(f"  Cost saving vs Big: {results['cost_saving_vs_always_big']*100:.2f}%")
    print(f"  Recall (catching Big-needed): {results['recall']*100:.2f}%")
    print(f"  Precision: {results['precision']*100:.2f}%")
    print(f"  F1 Score: {results['f1']*100:.2f}%")
    print(f"  False Negatives: {results['false_negatives']}")
    print(f"  False Positives: {results['false_positives']}")
    print(f"  Parameters: {extra_info['num_params']/1e6:.1f}M")
    print(f"  Train loss: {extra_info['train_loss']:.4f}")


def print_comparison(roberta_results: Dict, electra_results: Dict):
    print(f"\nModel Comparison")
    
    acc_diff = roberta_results['cascade_acc'] - electra_results['cascade_acc']
    print(f"  Accuracy: RoBERTa {roberta_results['cascade_acc']*100:.2f}%, ELECTRA {electra_results['cascade_acc']*100:.2f}% (diff: {acc_diff*100:+.2f}%)")
    
    cost_diff = roberta_results['big_call_rate'] - electra_results['big_call_rate']
    print(f"  Big call rate: RoBERTa {roberta_results['big_call_rate']*100:.2f}%, ELECTRA {electra_results['big_call_rate']*100:.2f}% (diff: {cost_diff*100:+.2f}%)")
    
    print(f"  Oracle gap: RoBERTa {roberta_results['accuracy_gap_from_oracle']*100:.2f}%, ELECTRA {electra_results['accuracy_gap_from_oracle']*100:.2f}%")
    print(f"  Size: RoBERTa 125M, ELECTRA 14M (8.9× smaller)")


def main():
    import sys
    
    batch_size, gradient_accumulation = setup_environment()
    
    train_file = "gsm8k_train_small_Qwen_Qwen2.5-1.5B-Instruct_big_Qwen_Qwen2.5-7B-Instruct.jsonl"
    test_file = "gsm8k_test_small_Qwen_Qwen2.5-1.5B-Instruct_big_Qwen_Qwen2.5-7B-Instruct.jsonl"
    
    print("\nLoading data...")
    train_all, train_records, train_texts, train_labels = prepare_data(train_file, "train")
    test_all, test_records, test_texts, test_labels = prepare_data(test_file, "test")
    print(f"Train: {len(train_records)}/{len(train_all)}, Test: {len(test_records)}/{len(test_all)}")
    
    # Normal mode: train with defaults
    wandb_run = setup_wandb()
    
    print("\nTraining RoBERTa...")
    roberta_preds, roberta_info = train_and_predict(
        "roberta-base",
        train_texts,
        train_labels,
        test_texts,
        batch_size,
        gradient_accumulation,
    )
    
    print("\nEvaluating RoBERTa...")
    oracle_summary = load_oracle_summary()
    roberta_results = evaluate_cascade(test_all, test_records, roberta_preds, "roberta", oracle_summary, "test")
    print_results("RoBERTa-base", roberta_results, "125M", roberta_info)
    
    print("\nTraining ELECTRA...")
    electra_preds, electra_info = train_and_predict(
        "google/electra-small-discriminator",
        train_texts,
        train_labels,
        test_texts,
        batch_size,
        gradient_accumulation,
    )
    
    print("\nEvaluating ELECTRA...")
    electra_results = evaluate_cascade(test_all, test_records, electra_preds, "electra", oracle_summary, "test")
    print_results("ELECTRA-small", electra_results, "14M", electra_info)
    
    # Question-only experiment
    print("\n" + "="*80)
    print("Question-Only Experiment (RoBERTa)")
    print("="*80)
    
    train_all_qonly, train_records_qonly, train_texts_qonly, train_labels_qonly = prepare_data(train_file, "train", question_only=True)
    test_all_qonly, test_records_qonly, test_texts_qonly, test_labels_qonly = prepare_data(test_file, "test", question_only=True)
    
    print("\nTraining RoBERTa (Question-Only)...")
    qonly_preds, qonly_info = train_and_predict(
        "roberta-base",
        train_texts_qonly,
        train_labels_qonly,
        test_texts_qonly,
        batch_size,
        gradient_accumulation,
    )
    
    print("\nEvaluating RoBERTa (Question-Only)...")
    question_only_results = evaluate_cascade(test_all_qonly, test_records_qonly, qonly_preds, "roberta_qonly", oracle_summary, "test")
    print_results("RoBERTa-base (Question-Only)", question_only_results, "125M", qonly_info)
    
    print("\n" + "="*80)
    print("Comparison: With Answer vs Question-Only")
    print("="*80)
    acc_diff = roberta_results['cascade_acc'] - question_only_results['cascade_acc']
    print(f"  Cascade accuracy:")
    print(f"    With Answer: {roberta_results['cascade_acc']*100:.2f}%")
    print(f"    Question-Only: {question_only_results['cascade_acc']*100:.2f}%")
    print(f"    Difference: {acc_diff*100:+.2f}% ({'Better' if acc_diff > 0 else 'Worse'} with answer)")
    
    cost_diff = roberta_results['big_call_rate'] - question_only_results['big_call_rate']
    print(f"  Big call rate:")
    print(f"    With Answer: {roberta_results['big_call_rate']*100:.2f}%")
    print(f"    Question-Only: {question_only_results['big_call_rate']*100:.2f}%")
    print(f"    Difference: {cost_diff*100:+.2f}%")
    
    # Print model comparison
    print_comparison(roberta_results, electra_results)
    
    # Save results
    results_dict = {
        "timestamp": datetime.now().isoformat(),
        "baselines": {
            "always_small": {
                "accuracy": roberta_results["always_small_acc"],
                "cost": 0.0,
            },
            "always_big": {
                "accuracy": roberta_results["always_big_acc"],
                "cost": 1.0,
            },
            "oracle": {
                "accuracy": roberta_results["oracle_acc"],
                "cost": roberta_results["oracle_big_rate"],
            }
        },
        "roberta": {
            "model": "roberta-base",
            "params": "125M",
            "metrics": roberta_results,
            "training": roberta_info,
        },
        "electra": {
            "model": "google/electra-small-discriminator",
            "params": "14M",
            "metrics": electra_results,
            "training": electra_info,
        },
        "roberta_question_only": {
            "model": "roberta-base",
            "params": "125M",
            "input_type": "question_only",
            "metrics": question_only_results,
            "training": qonly_info,
        } if question_only_results and qonly_info else None
    }
    
    # Remove any keys from results_dict whose values are None
    results_dict = {k: v for k, v in results_dict.items() if v is not None}
    
    output_file = "cascade_router_results_baseline.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_file}")
    
    if wandb_run:
        table_data = [
            ["Always Small", f"{roberta_results['always_small_acc']*100:.1f}%", "0%", "-", "-", "-", "-"],
            ["Always Big", f"{roberta_results['always_big_acc']*100:.1f}%", "100%", "-", "-", "-", "-"],
            ["Oracle", f"{roberta_results['oracle_acc']*100:.1f}%", f"{roberta_results['oracle_big_rate']*100:.1f}%", "0%", "-", "-", "-"],
            ["RoBERTa", f"{roberta_results['cascade_acc']*100:.1f}%", f"{roberta_results['big_call_rate']*100:.1f}%", f"{roberta_results['accuracy_gap_from_oracle']*100:.1f}%", "125M", f"{roberta_results['recall']*100:.1f}%", f"{roberta_results['false_negatives']}"],
            ["ELECTRA", f"{electra_results['cascade_acc']*100:.1f}%", f"{electra_results['big_call_rate']*100:.1f}%", f"{electra_results['accuracy_gap_from_oracle']*100:.1f}%", "14M", f"{electra_results['recall']*100:.1f}%", f"{electra_results['false_negatives']}"],
        ]
        if question_only_results:
            table_data.append([
                "RoBERTa (Q-Only)", 
                f"{question_only_results['cascade_acc']*100:.1f}%", 
                f"{question_only_results['big_call_rate']*100:.1f}%", 
                f"{question_only_results['accuracy_gap_from_oracle']*100:.1f}%", 
                "125M",
                f"{question_only_results['recall']*100:.1f}%",
                f"{question_only_results['false_negatives']}"
            ])
        
        summary_table = wandb.Table(
            columns=["Model", "Accuracy", "Cost", "Oracle Gap", "Size", "Recall", "FN"],
            data=table_data
        )
        wandb.log({"results_summary": summary_table})
        wandb.finish()
        print("WandB run finished")


if __name__ == "__main__":
    main()