# IIT Jodhpur MLOps End-to-End Pipeline (Group 14)

An end-to-end, production-style MLOps pipeline for automated text emotion classification. Built with Hugging Face, Kaggle, Weights & Biases, Docker, and GitHub Actions.

## 👥 Group Members & Contributions
* **Zeeshan Akhtar** (G25AIT2135) - Task 1 & 2 Setup, Repository Architecture
* **Nikunj R Patel** (G25AIT2072) - Task 3 & 4 Hugging Face Model Selection & Loader ; Kaggle Training Pipelines & W&B Tracking
* **Rodosi Biswas** (G25AIT2088) - Task 5 & 6 Hugging Face Hub Model Deployment, Docker Containerization for Inference
* **Sharvan Vittala** (Roll No) - [Contribution]

## 🛠️ Project Architecture
- **Modality:** Text
- **Dataset:** `dair-ai/emotion` (Text Emotion Dataset)
- **Model Architecture:** `distilbert-base-uncased` (~260MB base)

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
│   ├── prepare_data.py   # Production data cleaning & normalization script
│   └── inference.py      # Core inference execution logic (pulls from HF Hub)
├── .gitignore            # Git exclusion rules
├── Dockerfile            # Container deployment blueprint
├── LICENSE               # Open-source project license
├── README.md             # Project documentation and onboarding guide
└── requirements.txt      # System-wide Python package dependencies
```

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
