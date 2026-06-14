FROM python:3.10-slim

WORKDIR /app

# HF model repo to pull at inference time. Override at build with:
#   docker build --build-arg HF_MODEL_NAME=user/model -t mlops-a3-inference .
ARG HF_MODEL_NAME=zeeshan-hf/mlops-emotion-distilbert-group14
ENV HF_MODEL_NAME=${HF_MODEL_NAME}

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/

EXPOSE 8000
EXPOSE 8501

CMD uvicorn src.api:app --host 0.0.0.0 --port 8000 & streamlit run src/ui.py --server.port 8501 --server.address 0.0.0.0