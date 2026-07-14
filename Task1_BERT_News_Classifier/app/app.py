"""
Gradio app for live interaction with the fine-tuned BERT news topic classifier.
Run with: python app/app.py
"""

import gradio as gr
from transformers import pipeline

MODEL_DIR = "../model/final"
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

classifier = pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR, top_k=None)


def predict(headline: str):
    if not headline.strip():
        return {}
    results = classifier(headline)[0]
    return {LABEL_NAMES[int(r["label"].split("_")[-1])] if r["label"].startswith("LABEL_") else r["label"]: r["score"] for r in results}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(label="News Headline", placeholder="e.g. Federal Reserve raises interest rates again"),
    outputs=gr.Label(num_top_classes=4, label="Predicted Topic"),
    title="News Topic Classifier (BERT)",
    description="Fine-tuned bert-base-uncased on the AG News dataset. Classifies headlines into World, Sports, Business, or Sci/Tech.",
    examples=[
        "Manchester United wins dramatic final in extra time",
        "Apple unveils new chip with record-breaking performance",
        "Stocks tumble as inflation fears grip Wall Street",
        "UN Security Council meets over escalating border conflict",
    ],
)

if __name__ == "__main__":
    demo.launch()
