"""
Task 5: Auto Tagging Support Tickets Using LLM

Uses a zero-shot LLM (BART-MNLI, via Hugging Face zero-shot-classification pipeline)
to tag free-text support tickets with the top 3 most probable categories, driven purely
by prompt/label engineering (no fine-tuning required). Also compares a plain zero-shot
prompt against a few-shot-style prompt (label descriptions enriched with examples).
"""

import json
import random

import pandas as pd
from datasets import load_dataset
from transformers import pipeline

N_SAMPLES = 12
RANDOM_SEED = 42

CANDIDATE_LABELS = [
    "Technical Support",
    "Product Support",
    "Customer Service",
    "IT Support",
    "Billing and Payments",
    "Returns and Exchanges",
    "Service Outages and Maintenance",
    "Sales and Pre-Sales",
    "Human Resources",
    "General Inquiry",
]

# Few-shot-style hypothesis template gives the zero-shot NLI model more context per label.
ZERO_SHOT_TEMPLATE = "This support ticket is about {}."
FEW_SHOT_TEMPLATE = (
    "Considering common examples such as billing disputes, login issues, refund requests, "
    "and outage reports, this support ticket is best categorized as {}."
)


def load_sample():
    ds = load_dataset("Tobi-Bueck/customer-support-tickets", split="train")
    ds = ds.filter(lambda x: x["language"] == "en")
    ds = ds.shuffle(seed=RANDOM_SEED).select(range(N_SAMPLES))
    return ds


def tag_tickets(classifier, texts, template):
    results = []
    for text in texts:
        out = classifier(text, CANDIDATE_LABELS, hypothesis_template=template, multi_label=True)
        top3 = list(zip(out["labels"][:3], [round(s, 4) for s in out["scores"][:3]]))
        results.append(top3)
    return results


def main():
    print("Loading sample of English support tickets...")
    ds = load_sample()
    texts = [f"{row['subject']}. {row['body']}"[:1000] for row in ds]
    true_queue = ds["queue"]

    print("Loading zero-shot classification pipeline (facebook/bart-large-mnli)...")
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    print("Tagging with plain zero-shot prompt...")
    zero_shot_preds = tag_tickets(classifier, texts, ZERO_SHOT_TEMPLATE)

    print("Tagging with few-shot-enriched prompt...")
    few_shot_preds = tag_tickets(classifier, texts, FEW_SHOT_TEMPLATE)

    def top1_accuracy(preds, truth):
        correct = sum(1 for p, t in zip(preds, truth) if p[0][0] == t)
        return correct / len(truth)

    zero_shot_acc = top1_accuracy(zero_shot_preds, true_queue)
    few_shot_acc = top1_accuracy(few_shot_preds, true_queue)

    print(f"\nZero-shot top-1 accuracy vs. ground-truth queue: {zero_shot_acc:.2%}")
    print(f"Few-shot-style top-1 accuracy vs. ground-truth queue: {few_shot_acc:.2%}")

    output_rows = []
    for text, true_label, zs, fs in zip(texts, true_queue, zero_shot_preds, few_shot_preds):
        output_rows.append({
            "ticket_excerpt": text[:200],
            "true_queue": true_label,
            "zero_shot_top3": zs,
            "few_shot_top3": fs,
        })

    df = pd.DataFrame(output_rows)
    df.to_csv("tagging_results.csv", index=False)

    summary = {
        "zero_shot_top1_accuracy": zero_shot_acc,
        "few_shot_top1_accuracy": few_shot_acc,
        "n_samples": N_SAMPLES,
        "candidate_labels": CANDIDATE_LABELS,
    }
    with open("eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved detailed predictions to tagging_results.csv")
    print("Saved summary metrics to eval_results.json")


if __name__ == "__main__":
    main()
