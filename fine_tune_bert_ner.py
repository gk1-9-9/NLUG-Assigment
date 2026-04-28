"""
Fine-tune BERT model on CoNLL-2003 NER dataset
This script demonstrates how to fine-tune a pre-trained BERT model for NER
"""

import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer
)
from seqeval.metrics import classification_report
from seqeval.scheme import IOB2
import torch

def load_conll_dataset():
    """Load CoNLL-2003 dataset from Hugging Face"""
    print("Loading CoNLL-2003 dataset...")
    # eriktks/conll2003 is a Parquet-based mirror compatible with datasets>=3.0
    sources = ["eriktks/conll2003", "conll2003"]
    for source in sources:
        try:
            dataset = load_dataset(source)
            print(f"Dataset loaded from '{source}': {dataset}")
            print(f"Train size: {len(dataset['train'])}")
            print(f"Validation size: {len(dataset['validation'])}")
            print(f"Test size: {len(dataset['test'])}")
            return dataset
        except Exception as e:
            print(f"Could not load from '{source}': {e}")
    print("\nAll dataset sources failed. Skipping dataset loading.")
    print("The training code is ready to use once the dataset is available.")
    return None

def tokenize_and_align_labels(examples, tokenizer, label_all_tokens=True):
    """Tokenize and align labels with tokens"""
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
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
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p, label_list):
    """Compute metrics for evaluation"""
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

    report = classification_report(true_labels, true_predictions, mode='strict', scheme=IOB2, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(true_labels, true_predictions, mode='strict', scheme=IOB2))

    return {
        "precision": report["macro avg"]["precision"],
        "recall": report["macro avg"]["recall"],
        "f1": report["macro avg"]["f1-score"],
    }

def fine_tune_bert():
    """Main function to fine-tune BERT for NER"""
    print("=" * 60)
    print("Fine-tuning BERT for Named Entity Recognition")
    print("=" * 60)
    
    # Load dataset
    dataset = load_conll_dataset()

    # Get label list - use defaults if dataset unavailable
    if dataset is not None:
        features = dataset["train"].features
        label_list = features["ner_tags"].feature.names
    else:
        print("\nUsing default CoNLL-2003 labels for demonstration.")
        label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']

    print(f"\nLabel list: {label_list}")
    print(f"Number of labels: {len(label_list)}")
    
    # Load tokenizer and model
    model_name = "bert-base-cased"
    print(f"\nLoading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_list)
    )
    
    # Tokenize dataset (only when data is available)
    tokenized_datasets = None
    if dataset is not None:
        print("\nTokenizing dataset...")
        tokenized_datasets = dataset.map(
            lambda x: tokenize_and_align_labels(x, tokenizer),
            batched=True,
            remove_columns=dataset["train"].column_names
        )
    else:
        print("\nSkipping tokenization - dataset not loaded.")

    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./bert-ner-finetuned",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_ratio=0.1,
        push_to_hub=False,
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )
    
    # Initialize trainer (only when tokenized data is available)
    trainer = None
    if tokenized_datasets is not None:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=lambda p: compute_metrics(p, label_list)
        )
        print("\nTrainer initialised and ready.")
        print("Uncomment trainer.train() to start actual training.")
    else:
        print("\nTrainer not initialised - no tokenized dataset available.")

    print("\n" + "=" * 60)
    print("Training setup complete!")
    print("=" * 60)

    return trainer, tokenizer, label_list

def test_fine_tuned_model(trainer, tokenizer, label_list):
    """Test the fine-tuned model on sample text"""
    print("\n" + "=" * 60)
    print("Testing Fine-tuned Model")
    print("=" * 60)
    
    sample_text = "Tesla opened a new Gigafactory in Austin, Texas, investing over $5 billion in the facility."
    
    tokens = tokenizer(sample_text, return_tensors="pt")
    tokens = {k: v.to(trainer.model.device) for k, v in tokens.items()}

    with torch.no_grad():
        outputs = trainer.model(**tokens)
    
    predictions = outputs.logits.argmax(dim=-1)
    
    # Convert to labels
    tokens_list = tokenizer.convert_ids_to_tokens(tokens["input_ids"][0])
    predictions_list = [label_list[p] for p in predictions[0]]
    
    print(f"\nText: {sample_text}")
    print("\nToken-level predictions:")
    for token, label in zip(tokens_list, predictions_list):
        print(f"  {token:15} | {label}")
    
    # Group entities
    entities = []
    current_entity = None
    current_tokens = []
    
    for token, label in zip(tokens_list, predictions_list):
        if label == "O":
            if current_entity:
                entities.append((current_entity, "".join(current_tokens)))
                current_entity = None
                current_tokens = []
        elif label.startswith("B-"):
            if current_entity:
                entities.append((current_entity, "".join(current_tokens)))
            current_entity = label[2:]
            current_tokens = [token.replace("##", "")]
        elif label.startswith("I-"):
            if current_entity:
                current_tokens.append(token.replace("##", ""))
            else:
                current_entity = label[2:]
                current_tokens = [token.replace("##", "")]
    
    if current_entity:
        entities.append((current_entity, "".join(current_tokens)))
    
    print("\nExtracted entities:")
    for entity_type, entity_text in entities:
        print(f"  - {entity_text:20} | {entity_type}")

if __name__ == "__main__":
    trainer, tokenizer, label_list = fine_tune_bert()
    
    # Uncomment to test the model after training
    # test_fine_tuned_model(trainer, tokenizer, label_list)
