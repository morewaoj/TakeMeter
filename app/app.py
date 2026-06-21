from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "distilbert-takemeter"
LABELS = ["analysis", "hot_take", "reaction_noise"]


tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()


def classify_post(text: str) -> tuple[str, float]:
    text = text.strip()
    if not text:
        return "Enter an NBA post to classify.", 0.0

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)[0]

    predicted_id = int(torch.argmax(probabilities).item())
    label = model.config.id2label.get(predicted_id, LABELS[predicted_id])
    confidence = float(probabilities[predicted_id].item())
    return label, confidence


demo = gr.Interface(
    fn=classify_post,
    inputs=gr.Textbox(label="NBA post", lines=4, placeholder="Paste an NBA comment here..."),
    outputs=[
        gr.Textbox(label="Predicted label"),
        gr.Number(label="Confidence score", precision=4),
    ],
    title="TakeMeter",
    description="Classify NBA discourse as analysis, hot_take, or reaction_noise.",
    allow_flagging="never",
)


if __name__ == "__main__":
    demo.launch()
