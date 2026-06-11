"""Evaluate the fine-tuned emotion model on the held-out test split.

Produces the numbers behind the Task 8 comparison table: overall accuracy,
weighted F1 (handles the class imbalance in the emotion dataset), macro F1
(treats every emotion equally), a per-class report and a confusion matrix.

Usage:
    python notebooks/sharvan/evaluate_model.py
    MODEL_ID=<hf-username>/my-model python notebooks/sharvan/evaluate_model.py
"""
import os
import json

from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import pipeline

# Default to the team model; override with the MODEL_ID env var to test your own.
MODEL_ID = os.environ.get("MODEL_ID", "zeeshan-hf/mlops-emotion-distilbert-group14")

# id2label.json lives at the repo root under data/ - resolve it relative to this file.
HERE = os.path.dirname(os.path.abspath(__file__))
MAPPING_PATH = os.path.join(HERE, "..", "..", "data", "id2label.json")


def load_id2label():
    with open(MAPPING_PATH) as f:
        return {int(k): v for k, v in json.load(f).items()}


def to_label_id(raw_label, id2label):
    """Pipeline may return 'LABEL_3' or a readable name depending on the repo config."""
    if raw_label.startswith("LABEL_"):
        return int(raw_label.split("_")[1])
    name2id = {v: k for k, v in id2label.items()}
    return name2id.get(raw_label, -1)


def main():
    id2label = load_id2label()
    label_names = [id2label[i] for i in sorted(id2label)]

    print(f"Loading test split of dair-ai/emotion and model '{MODEL_ID}'...")
    test = load_dataset("dair-ai/emotion", split="test")
    classifier = pipeline("text-classification", model=MODEL_ID, truncation=True)

    preds = []
    for out in classifier(test["text"], batch_size=32):
        preds.append(to_label_id(out["label"], id2label))
    gold = test["label"]

    acc = accuracy_score(gold, preds)
    f1_weighted = f1_score(gold, preds, average="weighted")
    f1_macro = f1_score(gold, preds, average="macro")

    print("\n==== Test-set results ====")
    print(f"Samples      : {len(gold)}")
    print(f"Accuracy     : {acc:.4f}")
    print(f"F1 (weighted): {f1_weighted:.4f}")
    print(f"F1 (macro)   : {f1_macro:.4f}")

    print("\nPer-class report:")
    print(classification_report(gold, preds, target_names=label_names, digits=4))

    print("Confusion matrix (rows = true, cols = predicted):")
    print(label_names)
    print(confusion_matrix(gold, preds))


if __name__ == "__main__":
    main()
