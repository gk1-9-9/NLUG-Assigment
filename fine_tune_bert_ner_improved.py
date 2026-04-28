"""
Improved NER fine-tuning: RoBERTa-base with tuned hyperparameters.
Changes vs fine_tune_bert_ner.py:
  - model: roberta-base  (vs bert-base-cased)
  - label_all_tokens: False  (vs True)
  - learning_rate: 3e-5  (vs 2e-5)
  - num_train_epochs: 5  (vs 3)
  - batch_size: 32  (vs 16)
"""

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
import torch


def load_conll_dataset():
    """Load CoNLL-2003 from tner/conll2003 (plain JSONL, no loading script)."""
    print("Loading CoNLL-2003 dataset...")
    try:
        import json
        from huggingface_hub import hf_hub_download
        from datasets import Dataset, DatasetDict

        splits = {
            "train":      hf_hub_download("tner/conll2003", "dataset/train.json", repo_type="dataset"),
            "validation": hf_hub_download("tner/conll2003", "dataset/valid.json",  repo_type="dataset"),
            "test":       hf_hub_download("tner/conll2003", "dataset/test.json",   repo_type="dataset"),
        }

        def read_jsonl(path):
            with open(path) as f:
                records = [json.loads(line) for line in f]
            return {
                "tokens":   [r["tokens"] for r in records],
                "ner_tags": [r["tags"]   for r in records],
            }

        dataset = DatasetDict({s: Dataset.from_dict(read_jsonl(p)) for s, p in splits.items()})
        print(f"Dataset loaded: {dataset}")
        print(f"Train: {len(dataset['train'])}  Val: {len(dataset['validation'])}  Test: {len(dataset['test'])}")
        return dataset
    except Exception as e:
        print(f"Could not load dataset: {e}")
    print("All sources failed. Using default labels only.")
    return None


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)  # mask continuation sub-tokens
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(p, label_list):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    report = classification_report(
        true_labels, true_predictions, mode="strict", scheme=IOB2, output_dict=True
    )
    print("\nClassification Report:")
    print(classification_report(true_labels, true_predictions, mode="strict", scheme=IOB2))
    return {
        "precision": report["macro avg"]["precision"],
        "recall":    report["macro avg"]["recall"],
        "f1":        report["macro avg"]["f1-score"],
    }


def fine_tune_roberta():
    print("=" * 60)
    print("Fine-tuning RoBERTa for Named Entity Recognition (Improved)")
    print("=" * 60)

    dataset = load_conll_dataset()

    # tner/conll2003 label order (from dataset/label.json)
    label_list = ["O", "B-ORG", "B-MISC", "B-PER", "I-PER", "B-LOC", "I-ORG", "I-MISC", "I-LOC"]

    print(f"\nLabel list: {label_list}")
    print(f"Number of labels: {len(label_list)}")

    model_name = "roberta-base"
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list),
    )

    tokenized_datasets = None
    if dataset is not None:
        print("\nTokenizing dataset...")
        tokenized_datasets = dataset.map(
            lambda x: tokenize_and_align_labels(x, tokenizer),
            batched=True,
            remove_columns=dataset["train"].column_names,
        )
    else:
        print("\nSkipping tokenization - dataset not loaded.")

    data_collator = DataCollatorForTokenClassification(tokenizer)

    training_args = TrainingArguments(
        output_dir="./roberta-ner-finetuned",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        push_to_hub=False,
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
    )

    trainer = None
    if tokenized_datasets is not None:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda p: compute_metrics(p, label_list),
        )

        print("\nStarting training...")
        trainer.train()

        print("\nEvaluating on test set...")
        test_results = trainer.evaluate(tokenized_datasets["test"])
        print("\n" + "=" * 60)
        print("IMPROVED MODEL - TEST RESULTS")
        print("=" * 60)
        print(f"  Precision : {test_results['eval_precision']:.4f}")
        print(f"  Recall    : {test_results['eval_recall']:.4f}")
        print(f"  F1        : {test_results['eval_f1']:.4f}")
        print("=" * 60)
    else:
        print("\nTrainer not initialised - no tokenized dataset available.")

    return trainer, tokenizer, label_list


if __name__ == "__main__":
    fine_tune_roberta()
