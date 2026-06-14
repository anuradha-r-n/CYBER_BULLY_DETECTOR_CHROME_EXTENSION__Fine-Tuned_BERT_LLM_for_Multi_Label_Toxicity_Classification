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
device = torch.device("cpu")

# ----------------------------
# Load ON START (IMPORTANT FIX)
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)
model.to(device)
model.eval()

class InputText(BaseModel):
    text: str

label_cols = [
    "toxic", "obscene", "insult",
    "severe_toxic", "identity_hate", "threat"
]

index_to_label = {i: label for i, label in enumerate(label_cols)}

@app.get("/")
def root():
    return {"status": "running"}

@app.post("/predict")
def predict(input_data: InputText):

    text = input_data.text

    if not text.strip():
        return {"predicted_labels": []}

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
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

    return {"predicted_labels": predicted_labels}
