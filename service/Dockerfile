# Build from the repository root:  docker build -f service/Dockerfile -t cmdparser:1.0 .
FROM python:3.11.9-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 TOKENIZERS_PARALLELISM=false
WORKDIR /app
# CPU-only PyTorch wheel (no CUDA libraries) keeps the image small; no model weights are copied.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu torch==2.5.1
COPY service/requirements.txt .
RUN pip install -r requirements.txt
COPY service/app.py service/schema.json ./
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
