# IIT Jodhpur MLOps End-to-End Pipeline (Group 14)

An end-to-end, production-style MLOps pipeline for automated text emotion classification. Built with Hugging Face, Kaggle, Weights & Biases, Docker, and GitHub Actions.

## 👥 Group Members & Contributions
* **Zeeshan Akhtar** (G25AIT2135) - Task 1 & 2 Setup, Repository Architecture
* **Nikunj R Patel** (G25AIT2072) - Task 3 & 4 Hugging Face Model Selection & Loader ; Kaggle Training Pipelines & W&B Tracking
* **Member 3 Name** (Roll No) - [Contribution]
* **Member 4 Name** (Roll No) - [Contribution]

## 🛠️ Project Architecture
- **Modality:** Text
- **Dataset:** `dair-ai/emotion` (Text Emotion Dataset)
- **Model Architecture:** `distilbert-base-uncased` (~260MB base)

## 📁 Repository Directory Structure

```text
mlops-pipeline-group14/
├── .github/workflows/
│   ├── ci.yml            # Code quality check workflow
│   └── inference.yml     # Automated inference workflow
├── src/
│   ├── prepare_data.py   # Data cleaning & normalization script
│   └── inference.py      # Inference/prediction execution logic
├── data/
│   └── id2label.json     # Label to text configuration mapping
├── .gitignore            # Git exclusion rules
├── Dockerfile            # Container deployment blueprint
├── LICENSE               # Open-source license
├── README.md             # Project documentation
└── requirements.txt      # System package dependencies
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
- [ ] **Task 5:** Hugging Face Hub Model Deployment
- [ ] **Task 6:** Docker Containerization for Inference
- [ ] **Task 7:** GitHub Actions Automation Workflows
- [ ] **Task 8:** Show All Experiments on W&B
- [ ] **Report (PDF):** Final Documentation Compilation
