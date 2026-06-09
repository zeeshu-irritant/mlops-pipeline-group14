# 🧠 MLOps Emotion Classification Pipeline (Group 14)

[![CI - Code Quality](https://github.com/zeeshu-irritant/mlops-pipeline-group14/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/mlops-pipeline-group14/actions/workflows/ci.yml)
[![CD - Automated Inference](https://github.com/zeeshu-irritant/mlops-pipeline-group14/actions/workflows/inference.yml/badge.svg)](https://github.com/your-username/mlops-pipeline-group14/actions/workflows/inference.yml)

## 📌 Project Overview
This repository contains a complete, end-to-end Machine Learning Operations (MLOps) pipeline. Our team fine-tuned a **DistilBERT** model to classify text into human emotions (Joy, Anger, Surprise, etc.). We then transitioned the model from raw experimental notebooks into a production-ready, containerized microservice architecture featuring an automated CI/CD pipeline.

---

## 🚀 The Development Journey (How It Works)

Our project was built systematically in distinct stages to mirror industry-standard MLOps practices:

1. **Experimentation & Training (Kaggle):** We used Kaggle's GPU environments to clean the dataset and fine-tune a pre-trained DistilBERT model.
2. **Experiment Tracking (Weights & Biases):** All hyperparameter sweeps, loss metrics, and hardware utilization stats were automatically logged to a centralized W&B team dashboard.
3. **Model Registry (Hugging Face):** The highest-performing model weights were pushed to the Hugging Face Model Hub, completely decoupling our model storage from our codebase.
4. **Microservice Containerization (Docker):** We built a dual-service Docker container. It pulls the model from Hugging Face and simultaneously spins up:
   * **FastAPI Backend:** A high-performance REST API handling the inference logic.
   * **Streamlit Frontend:** An interactive web dashboard for user input and visualization.
5. **Continuous Integration & Deployment (GitHub Actions):** Every push to `develop` or `main` triggers automated Python linting (Flake8) and builds the Docker container in the cloud to run a "smoke test," ensuring production code is never broken.

---

## 👥 Group Members & Contributions
* **Zeeshan Akhtar** (G25AIT2135)
* **Nikunj R Patel** (G25AIT2072)
* **Rodosi Biswas** (G25AIT2088)
* **Sharvan Vittala** (Roll No)

---

## 🛠️ Project Architecture
- **Modality:** Text
- **Dataset:** `dair-ai/emotion` (Text Emotion Dataset)
- **Model Architecture:** `distilbert-base-uncased` (~260MB base)

---

## 📂 Repository Directory Structure

```text
mlops-pipeline-group14/
├── .github/workflows/
│   ├── ci.yml            # Code quality check workflow (GitHub Actions)
│   └── inference.yml     # Automated inference workflow
├── data/
│   └── id2label.json     # Label-to-text configuration mapping
├── notebooks/            # Team experimentation & model tracking
│   ├── nikunj/           # Experimentation sandbox for Team Member Rodosi
│   ├── rodosi/           # Experimentation sandbox for Team Member Rodosi
│   └── sharvan/          # Experimentation sandbox for Team Member Sharvan
│   └── zeeshan/          # Experimentation sandbox for Team Member Zeeshan
├── src/                  # Production application codebase
│   ├── api.py            # FastAPI REST backend for model inference
│   ├── ui.py             # Streamlit interactive web frontend
│   ├── inference.py      # Core inference logic testing script (pulls from HF Hub)
│   └── prepare_data.py   # Data cleaning & normalization script
├── .gitignore            # Git exclusion rules
├── Dockerfile            # Container deployment blueprint
├── LICENSE               # Open-source project license
├── README.md             # Project documentation and onboarding guide
└── requirements.txt      # System-wide Python package dependencies
```
---

## 🚀 Getting Started

### 1. Clone the Repository & Environment Setup
```bash
git clone https://github.com/zeeshu-irritant/mlops-pipeline-group14.git
cd mlops-pipeline-group14
git checkout develop
```

### 2. Environment Activation & Dependencies
```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 3. Run Data Processing
To inspect dataset health metrics and run the cleaning rules locally:
```bash
python src/prepare_data.py
```

---

## 💻 How to Run the Application Locally
You can spin up the entire application using Docker. No local Python environments are required!

### 1. Build the Docker Image
```bash
docker build -t mlops-emotion-inference:latest .
```

### 2. Run the Container
```bash
docker run --rm \
  -p 8000:8000 \
  -p 8501:8501 \
  -e HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
  -e TRANSFORMERS_VERBOSITY=error \
  -e HF_HUB_VERBOSITY=error \
  mlops-emotion-inference:latest
```

### 3. Access the Interfaces
* **Web Dashboard (Streamlit):** Open your browser and go to `http://localhost:8501`
* **API Documentation (FastAPI):** Open your browser and go to `http://localhost:8000/docs`

---

## 👥 Team MLOps Onboarding Guide

To collaborate on experiments and ensure all logs sync to our centralized dashboards, every team member must configure their Kaggle environment using our shared team credentials.

### 1. Centralized Dashboards
* **Weights & Biases Project:** [Hugging Face | mlops-emotion-distilbert-group14](https://huggingface.co/zeeshan-hf/mlops-emotion-distilbert-group14)
* **Hugging Face Model Registry:** [WANDB.ai | mlops-emotion-classification](https://wandb.ai/zeeshu-irritant-prom-iit-rajasthan/mlops-emotion-classification)

### 2. Kaggle Environment Setup (Mandatory)
Before running any notebooks in the `notebooks/` directory, you must add our shared API keys to your personal Kaggle account. 

1. Open your notebook on Kaggle.
2. In the top menu, go to **Add-ons** -> **Secrets**.
3. Add the following two secrets exactly as written (retrieve the raw token values from our private team chat):
   * **Label:** `WANDB_API_KEY` | **Value:** *[Our shared W&B key]*
   * **Label:** `HF_TOKEN`      | **Value:** *[Our shared Hugging Face write token]*
4. Ensure the checkboxes next to both secrets are **checked** to attach them to your notebook session.
5. In the right-hand panel under **Session Options**, ensure **Internet on** is enabled.

### 3. Notebook Contribution Workflow
When saving your work from Kaggle back to GitHub:
1. Ensure your notebook is named following our convention: `notebooks/[your_name]/expX_[description].ipynb`.
2. Use the **Save to GitHub** option under the three dots (`...`) menu in Kaggle.
3. Target the active development branch (e.g., `feature/task4-kaggle-notebooks`), **never** push directly to `main` or `develop`.

---

## 📈 Pipeline Progress Checklist
- [x] **Task 1:** Set Up GitHub Repository & Branch Protection
- [x] **Task 2:** Data Cleaning & Normalisation Module
- [x] **Task 3:** Hugging Face Model Selection & Loader
- [x] **Task 4:** Kaggle Training Pipelines & W&B Tracking
- [x] **Task 5:** Hugging Face Hub Model Deployment
- [x] **Task 6:** Docker Containerization for Inference
- [ ] **Task 7:** GitHub Actions Automation Workflows
- [ ] **Task 8:** Show All Experiments on W&B
- [ ] **Report (PDF):** Final Documentation Compilation

---
