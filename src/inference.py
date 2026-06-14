import os
import json
from transformers import pipeline

MODEL_ID = os.environ.get("HF_MODEL_NAME", "zeeshan-hf/mlops-emotion-distilbert-group14")
MAPPING_PATH = os.path.join("data", "id2label.json")

def load_mapping():
    # Load the local label mapping file
    if os.path.exists(MAPPING_PATH):
        with open(MAPPING_PATH, "r") as f:
            # Convert string keys back to integers for safety
            return {int(k): v for k, v in json.load(f).items()}
    else:
        print(f"Warning: Mapping file not found at {MAPPING_PATH}. Using default labels.")
        return None

def load_model():
    print(f"Loading model '{MODEL_ID}' from Hugging Face Hub...")
    # Load text classification pipeline
    return pipeline("text-classification", model=MODEL_ID)

def predict(classifier, text, label_mapping):
    print(f"\nAnalyzing text: '{text}'")
    result = classifier(text)
    prediction = result[0]
    
    raw_label = prediction['label']
    
    if raw_label.startswith("LABEL_") and label_mapping:
        label_id = int(raw_label.split("_")[1])
        readable_label = label_mapping.get(label_id, raw_label)
    else:
        readable_label = raw_label

    print(f"Predicted Emotion: {readable_label.upper()} (Confidence: {prediction['score']:.4f})")
    return prediction

if __name__ == "__main__":
    # The GitHub Actions inference workflow passes the text to classify via
    # the INPUT_TEXT env var. When it is absent (local runs / Docker smoke
    # test) we fall back to a few sample sentences.
    input_text = os.environ.get("INPUT_TEXT", "").strip()
    if input_text:
        texts = [input_text]
    else:
        texts = [
            "I am so incredibly proud of my team for finishing this MLOps project!",
            "The server crashed right before the deadline, and I am furious.",
            "I honestly didn't expect the accuracy to jump that high."
        ]

    # Load configuration mapping and model
    label_mapping = load_mapping()
    classifier = load_model()

    print("\n--- Starting Inference ---")
    for text in texts:
        predict(classifier, text, label_mapping)
    print("--- Inference Complete ---")