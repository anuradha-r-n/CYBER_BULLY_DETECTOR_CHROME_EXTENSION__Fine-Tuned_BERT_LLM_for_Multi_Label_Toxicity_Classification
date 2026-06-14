from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
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
# Lazy loading (IMPORTANT FIX)
# ----------------------------
tokenizer = None
model = None

def load_model():
    global tokenizer, model

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_id)

    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.eval()

    return tokenizer, model


# ----------------------------
# Health check endpoint
# ----------------------------
@app.get("/")
def root():
    return {"status": "running"}


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
# Prediction endpoint
# ----------------------------
@app.post("/predict")
async def predict(request: Request):

    tokenizer, model = load_model()

    data = await request.json()
    text = data.get("text", "")

    if not text.strip():
        return {"error": "No text provided"}

    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    # Move inputs to same device as model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Model inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.sigmoid(logits).squeeze().cpu().tolist()

    # Thresholding
    threshold = 0.5
    predicted_labels = [
        index_to_label[i]
        for i, p in enumerate(probs)
        if p >= threshold
    ]

    return {"predicted_labels": predicted_labels}
