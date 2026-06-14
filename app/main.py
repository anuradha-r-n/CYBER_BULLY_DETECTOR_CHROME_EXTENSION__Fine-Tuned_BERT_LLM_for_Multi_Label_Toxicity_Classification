from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Hugging Face config
# -------------------------
HF_TOKEN = os.getenv("HF_TOKEN")  # set in Render environment variables

API_URL = "https://api-inference.huggingface.co/models/anu111222/cyberbully-detector"

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# -------------------------
# Request schema
# -------------------------
class InputText(BaseModel):
    text: str


label_cols = [
    "toxic",
    "obscene",
    "insult",
    "severe_toxic",
    "identity_hate",
    "threat"
]


@app.get("/")
def root():
    return {"status": "running (HF API mode)"}


# -------------------------
# Call Hugging Face API
# -------------------------
@app.post("/predict")
def predict(data: InputText):

    if not data.text.strip():
        return {"predicted_labels": []}

    payload = {
        "inputs": data.text
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    result = response.json()

    # Handle HF response format
    if isinstance(result, dict) and "error" in result:
        return {"error": result["error"], "predicted_labels": []}

    try:
        scores = result[0]
    except:
        return {"predicted_labels": []}

    threshold = 0.5

    predicted_labels = [
        label_cols[i]
        for i, item in enumerate(scores)
        if item["score"] >= threshold
    ]

    return {
        "predicted_labels": predicted_labels
    }
