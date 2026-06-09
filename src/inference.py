import os
from transformers import pipeline

MODEL_ID = "zeeshan-hf/mlops-emotion-distilbert-group14"

def load_model():
    print(f"Loading model '{MODEL_ID}' from Hugging Face Hub...")
    emotion_classifier = pipeline("text-classification", model=MODEL_ID)
    return emotion_classifier

def predict(classifier, text):
    print(f"\nAnalyzing text: '{text}'")
    result = classifier(text)
    prediction = result[0]
    print(f"Predicted Emotion: {prediction['label'].upper()} (Confidence: {prediction['score']:.4f})")
    return prediction

if __name__ == "__main__":
    # Test strings for our container to evaluate
    sample_texts = [
        "I am so incredibly proud of my team for finishing this MLOps project!",
        "The server crashed right before the deadline, and I am furious.",
        "I honestly didn't expect the accuracy to jump that high."
    ]
    
    # Initialize the model
    classifier = load_model()
    
    # Run predictions
    print("\n--- Starting Inference ---")
    for text in sample_texts:
        predict(classifier, text)
    print("--- Inference Complete ---")