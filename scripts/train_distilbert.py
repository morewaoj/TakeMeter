#!/usr/bin/env python3
"""Fine-tune DistilBERT for the TakeMeter 3-class classifier.

Inputs:
- data/train.csv
- data/validation.csv
- data/test.csv

Outputs:
- results/finetuned_predictions.csv
- results/finetuned_metrics.md

This script trains only when executed directly. It does not create fake results.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_DIR = ROOT / "models" / "distilbert-takemeter"
PREDICTIONS_PATH = RESULTS_DIR / "finetuned_predictions.csv"
METRICS_PATH = RESULTS_DIR / "finetuned_metrics.md"

LABELS = ["analysis", "hot_take", "reaction_noise"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}


class WeightedLossTrainer(Trainer):
    """Trainer that applies class-weighted cross entropy during training."""

    def __init__(self, *args, class_weights: torch.Tensor, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fn(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def load_split(path: Path) -> Dataset:
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "text": row["text"],
                    "label": LABEL_TO_ID[row["label"]],
                    "gold_label": row["label"],
                }
            )
    return Dataset.from_list(rows)


def load_dataset() -> DatasetDict:
    return DatasetDict(
        {
            "train": load_split(DATA_DIR / "train.csv"),
            "validation": load_split(DATA_DIR / "validation.csv"),
            "test": load_split(DATA_DIR / "test.csv"),
        }
    )


def compute_class_weights(train_dataset: Dataset) -> torch.Tensor:
    train_labels = np.asarray(train_dataset["label"])
    class_counts = np.bincount(train_labels, minlength=len(LABELS))
    if np.any(class_counts == 0):
        missing = [LABELS[index] for index, count in enumerate(class_counts) if count == 0]
        raise ValueError(f"Cannot compute class weights; missing train labels: {missing}")

    total = class_counts.sum()
    weights = total / (len(LABELS) * class_counts)
    return torch.tensor(weights, dtype=torch.float32)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(ID_TO_LABEL.keys()),
        average="macro",
        zero_division=0,
    )
    return {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }


def calculate_metric_bundle(gold_ids: np.ndarray, predicted_ids: np.ndarray) -> dict:
    accuracy = accuracy_score(gold_ids, predicted_ids)
    precision, recall, f1, _ = precision_recall_fscore_support(
        gold_ids,
        predicted_ids,
        labels=list(ID_TO_LABEL.keys()),
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        gold_ids,
        predicted_ids,
        labels=list(ID_TO_LABEL.keys()),
        average="macro",
        zero_division=0,
    )
    matrix = confusion_matrix(gold_ids, predicted_ids, labels=list(ID_TO_LABEL.keys()))
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "matrix": matrix,
        "total": len(gold_ids),
    }


def write_predictions(test_dataset: Dataset, predicted_ids: np.ndarray) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with PREDICTIONS_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["text", "gold_label", "predicted_label", "correct"])
        writer.writeheader()
        for row, predicted_id in zip(test_dataset, predicted_ids):
            predicted_label = ID_TO_LABEL[int(predicted_id)]
            gold_label = row["gold_label"]
            writer.writerow(
                {
                    "text": row["text"],
                    "gold_label": gold_label,
                    "predicted_label": predicted_label,
                    "correct": str(predicted_label == gold_label).lower(),
                }
            )


def append_metric_section(lines: list[str], title: str, metrics: dict) -> None:
    lines.extend(
        [
            f"## {title}",
            "",
            f"Examples: {metrics['total']}",
            "",
            f"Accuracy: {metrics['accuracy']:.4f}",
            "",
            f"Macro F1: {metrics['macro_f1']:.4f}",
            "",
            "### Per-Class Metrics",
            "",
            "| Label | Precision | Recall | F1 |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for index, label in enumerate(LABELS):
        lines.append(
            f"| `{label}` | {metrics['precision'][index]:.4f} | "
            f"{metrics['recall'][index]:.4f} | {metrics['f1'][index]:.4f} |"
        )

    lines.extend(
        [
            "",
            "### Macro Averages",
            "",
            "| Precision | Recall | F1 |",
            "| ---: | ---: | ---: |",
            f"| {metrics['macro_precision']:.4f} | {metrics['macro_recall']:.4f} | {metrics['macro_f1']:.4f} |",
            "",
            "### Confusion Matrix",
            "",
            "Rows are gold labels; columns are predicted labels.",
            "",
            "| Gold \\ Predicted | analysis | hot_take | reaction_noise |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row_index, gold_label in enumerate(LABELS):
        counts = metrics["matrix"][row_index]
        lines.append(f"| `{gold_label}` | {counts[0]} | {counts[1]} | {counts[2]} |")
    lines.append("")


def write_metrics(validation_metrics: dict, test_metrics: dict, class_weights: torch.Tensor) -> None:
    lines = [
        "# Fine-Tuned DistilBERT Metrics",
        "",
        f"Base model: `{MODEL_NAME}`",
        "",
        "Loss: class-weighted cross entropy using weights computed from `data/train.csv` only.",
        "",
        "Dataset splits: `data/train.csv`, `data/validation.csv`, `data/test.csv`",
        "",
        "## Class Weights",
        "",
        "| Label | Weight |",
        "| --- | ---: |",
    ]
    for index, label in enumerate(LABELS):
        lines.append(f"| `{label}` | {float(class_weights[index]):.4f} |")
    lines.append("")

    append_metric_section(lines, "Validation Metrics", validation_metrics)
    append_metric_section(lines, "Final Test Metrics", test_metrics)

    METRICS_PATH.write_text("\n".join(lines))


def main() -> int:
    dataset = load_dataset()
    class_weights = compute_class_weights(dataset["train"])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True)

    tokenized = dataset.map(tokenize, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        remove_unused_columns=True,
        logging_dir=str(ROOT / "logs" / "distilbert"),
        logging_steps=10,
        report_to="none",
        seed=42,
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        class_weights=class_weights,
    )

    trainer.train()
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    validation_output = trainer.predict(tokenized["validation"])
    validation_predicted_ids = np.argmax(validation_output.predictions, axis=-1)
    validation_gold_ids = np.asarray(validation_output.label_ids)
    validation_metrics = calculate_metric_bundle(validation_gold_ids, validation_predicted_ids)

    test_output = trainer.predict(tokenized["test"])
    predicted_ids = np.argmax(test_output.predictions, axis=-1)
    gold_ids = np.asarray(test_output.label_ids)
    test_metrics = calculate_metric_bundle(gold_ids, predicted_ids)

    write_predictions(dataset["test"], predicted_ids)
    write_metrics(validation_metrics, test_metrics, class_weights)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
