import streamlit as st
import requests

# Page Configuration
st.set_page_config(page_title="Emotion Analyzer", page_icon="🧠", layout="centered")

st.title("🧠 Centralized Emotion Classifier")
st.write("Type a sentence below to see our group's fine-tuned model predict its underlying emotion.")

# Sidebar metadata for evaluation visibility
st.sidebar.header("🛠️ Pipeline Metadata")
st.sidebar.info("**Model:** DistilBERT-Emotion\n\n**Backend:** FastAPI\n\n**Frontend:** Streamlit")

# Text Input Area
user_input = st.text_area("Enter text to analyze:", placeholder="I'm absolutely thrilled with how this pipeline came together!")

# Define the FastAPI endpoint URL (local network communication)
API_URL = "http://localhost:8000/predict"

if st.button("Analyze Emotion", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Analyzing text..."):
            try:
                # Send HTTP POST request to the FastAPI container backend
                response = requests.post(API_URL, json={"text": user_input})
                
                if response.status_code == 200:
                    data = response.json()
                    emotion = data["emotion"]
                    confidence = data["confidence"]
                    
                    # Display Results using stylized metrics cards
                    st.success("Analysis Complete!")
                    col1, col2 = st.columns(2)
                    col1.metric(label="Predicted Emotion", value=emotion)
                    col2.metric(label="Confidence Score", value=f"{confidence * 100:.2f}%")
                    
                    # Add a nice visual progress bar representing the model's confidence
                    st.progress(confidence)
                else:
                    st.error(f"Backend API error: {response.text}")
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the FastAPI backend service. Is the backend running?")