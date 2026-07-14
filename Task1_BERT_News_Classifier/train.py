"""
Task 1: News Topic Classifier Using BERT
Fine-tunes bert-base-uncased on the AG News dataset to classify headlines
into 4 topic categories: World, Sports, Business, Sci/Tech.

Sized to run on CPU: uses a subset of AG News and a short training run.
Increase TRAIN_SUBSET / EPOCHS if you have a GPU available.
"""

import glob
import os

import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

MODEL_NAME = "bert-base-uncased"
OUTPUT_DIR = "./model"
TRAIN_SUBSET = 8000
EVAL_SUBSET = 1000
EPOCHS = 2
BATCH_SIZE = 16

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def main():
    print("Loading AG News dataset...")
    raw = load_dataset("fancyzhx/ag_news")
    train_ds = raw["train"].shuffle(seed=42).select(range(TRAIN_SUBSET))
    eval_ds = raw["test"].shuffle(seed=42).select(range(EVAL_SUBSET))

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABEL_NAMES)
    )

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    train_ds = train_ds.map(tokenize, batched=True)
    eval_ds = eval_ds.map(tokenize, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_metric.compute(predictions=preds, references=labels)
        f1 = f1_metric.compute(predictions=preds, references=labels, average="weighted")
        return {"accuracy": acc["accuracy"], "f1": f1["f1"]}

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    checkpoints = sorted(
        glob.glob(os.path.join(OUTPUT_DIR, "checkpoint-*")),
        key=lambda p: int(p.rsplit("-", 1)[-1]),
    )
    resume_from = checkpoints[-1] if checkpoints else None
    if resume_from:
        print(f"Resuming training from {resume_from}")

    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from)

    print("Final evaluation...")
    metrics = trainer.evaluate()
    print(metrics)

    print(f"Saving model to {OUTPUT_DIR}/final")
    trainer.save_model(f"{OUTPUT_DIR}/final")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")

    with open("eval_results.txt", "w") as f:
        f.write(str(metrics))


if __name__ == "__main__":
    main()
