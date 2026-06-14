from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_id = "anu111222/cyberbully-detector"

# ----------------------------
# Device setup (DO ONCE)
# ----------------------------
device = torch.device("cpu")  # safer for Render + local consistency

# ----------------------------
# Lazy loading
# ----------------------------
tokenizer = None
model = None

def load_model():
    global tokenizer, model

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.to(device)
        model.eval()

    return tokenizer, model


# ----------------------------
# Health check
# ----------------------------
@app.get("/")
def root():
    return {"status": "running"}


# ----------------------------
# Input schema (IMPORTANT FIX)
# ----------------------------
class InputText(BaseModel):
    text: str


# ----------------------------
# Label mapping
# ----------------------------
label_cols = [
    "toxic",
    "obscene",
    "insult",
    "severe_toxic",
    "identity_hate",
    "threat"
]

index_to_label = {i: label for i, label in enumerate(label_cols)}


# ----------------------------
# Prediction endpoint (FIXED)
# ----------------------------
@app.post("/predict")
def predict(input_data: InputText):

    text = input_data.text

    if not text.strip():
        return {"error": "No text provided", "predicted_labels": []}

    tokenizer, model = load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).squeeze().tolist()

    threshold = 0.5

    predicted_labels = [
        index_to_label[i]
        for i, p in enumerate(probs)
        if p >= threshold
    ]

    return {
        "predicted_labels": predicted_labels
    }
