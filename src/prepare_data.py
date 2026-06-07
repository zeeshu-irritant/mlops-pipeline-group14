import os
import re
import json
import pandas as pd
from datasets import load_dataset


def inspect_and_clean_data():
    print("TASK 2: Data Inspection Started...")

    # Load Dataset
    raw_dataset = load_dataset("dair-ai/emotion")

    # Convert to Pandas for robust cleaning operations
    train_df = pd.DataFrame(raw_dataset["train"])
    val_df = pd.DataFrame(raw_dataset["validation"])

    # Inspect Raw Data (Size, Structure, Distribution)
    print(f"Raw Train Size: {len(train_df)} rows")
    print(f"Raw Validation Size: {len(val_df)} rows")
    print("\nRaw Class Distribution (Train Split):")
    print(train_df["label"].value_counts())

    # Clean and Normalise Data
    def clean_text(text):
        if not isinstance(text, str):
            return ""
        text = text.lower()  # Lowercase text
        text = re.sub(
            r"[^\w\s]", "", text
        )  # Strip punctuation using regex
        text = text.strip()  # Remove trailing whitespaces
        return text

    print("\nExecuting Data Cleaning Pipeline...")
    for df in [train_df, val_df]:
        # Handle potential missing text values
        df.dropna(subset=["text"], inplace=True)
        # Apply standard normalisation
        df["text"] = df["text"].apply(clean_text)
        # Drop rows where text became empty after clearing punctuation
        df = df[df["text"] != ""]
        # Remove duplicate rows
        df.drop_duplicates(subset=["text"], inplace=True)

    print(f"Cleaned Train Size: {len(train_df)} rows")
    print(f"Cleaned Validation Size: {len(val_df)} rows")

    # Save Cleaned Splits Locally (Not committed to Git)
    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train_clean.csv", index=False)
    val_df.to_csv("data/val_clean.csv", index=False)
    print("Cleaned splits successfully exported to local 'data/' directory.")

    # Verify Label Mapping File Presence
    if os.path.exists("data/id2label.json"):
        print("Label mapping validation passed: 'data/id2label.json' exists.")
    else:
        print("Warning: Ensure 'data/id2label.json' is present.")


if __name__ == "__main__":
    inspect_and_clean_data()
