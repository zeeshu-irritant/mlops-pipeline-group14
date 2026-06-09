import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

# Initialize FastAPI app
app = FastAPI(title="Emotion Classification API", version="1.0")

MODEL_ID = "zeeshan-hf/mlops-emotion-distilbert-group14"
MAPPING_PATH = os.path.join("data", "id2label.json")

# Define the expected request payload structure
class TextRequest(BaseModel):
    text: str

# Global variables to hold the model and mapping
classifier = None
label_mapping = None

@app.on_event("startup")
def startup_event():
    global classifier, label_mapping
    print("Initializing system and loading assets...")
    
    # Load label mapping
    if os.path.exists(MAPPING_PATH):
        with open(MAPPING_PATH, "r") as f:
            label_mapping = {int(k): v for k, v in json.load(f).items()}
    
    # Load Hugging Face Pipeline
    classifier = pipeline("text-classification", model=MODEL_ID)
    print("Model successfully loaded and ready!")

@app.get("/")
def home():
    return {"status": "healthy", "model": MODEL_ID}

@app.post("/predict")
def predict_emotion(payload: TextRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=400, detail="Text string cannot be empty.")
    
    # Run inference
    result = classifier(payload.text)
    prediction = result[0]
    
    raw_label = prediction['label']
    confidence = prediction['score']
    
    # Apply human-readable mapping if applicable
    if raw_label.startswith("LABEL_") and label_mapping:
        label_id = int(raw_label.split("_")[1])
        readable_label = label_mapping.get(label_id, raw_label)
    else:
        readable_label = raw_label
        
    return {
        "text": payload.text,
        "emotion": readable_label.upper(),
        "confidence": float(confidence)
    }