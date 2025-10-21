FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app


RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*


COPY requirements.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt


COPY . .

EXPOSE 8000

# Start FastAPI server
CMD ["python", "-m", "src.retrieval_server", "--port", "8000"]