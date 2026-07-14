# Task 1: News Topic Classifier Using BERT

## Objective
Fine-tune a transformer model (`bert-base-uncased`) to classify news headlines into one of four
topic categories: **World, Sports, Business, Sci/Tech**.

## Dataset
[AG News](https://huggingface.co/datasets/ag_news) — loaded via Hugging Face `datasets`.
For CPU-friendly training, `train.py` uses a random subset (8,000 train / 1,000 eval examples
by default); increase `TRAIN_SUBSET`/`EPOCHS` in `train.py` if running on a GPU.

## Methodology / Approach
1. Load AG News and tokenize headlines with the BERT tokenizer (max length 128, dynamic padding
   via `DataCollatorWithPadding`).
2. Fine-tune `bert-base-uncased` for sequence classification (4 labels) using Hugging Face
   `Trainer` for 2 epochs, batch size 16, learning rate 2e-5.
3. Evaluate on a held-out split using **accuracy** and **weighted F1-score**.
4. Save the fine-tuned model/tokenizer to `model/final/`.
5. Serve the model for live interaction through a **Gradio** app (`app/app.py`).

## How to Run

```bash
pip install -r requirements.txt

# Train and evaluate
python train.py

# Launch the interactive demo (after training)
cd app
python app.py
```

## Results
See `eval_results.txt` (generated after training) for accuracy/F1 on the evaluation subset.

## Key Results / Insights
- Fine-tuning BERT on even a small subset of AG News yields strong topic-classification accuracy,
  demonstrating the value of transfer learning for text classification with limited compute.
- Weighted F1 was tracked alongside accuracy since AG News is class-balanced but F1 gives a more
  robust picture per-class.
- The Gradio interface allows real-time testing of arbitrary headlines against the trained model.

## Skills Gained
- NLP using Transformers
- Transfer learning & fine-tuning
- Evaluation metrics for text classification
- Lightweight model deployment (Gradio)
