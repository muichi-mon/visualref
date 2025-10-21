FROM python:3.10-slim

RUN apt-get update && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7862

CMD python -m src.demo.app_client \
    --client_config_path ${CLIENT_CONFIG_PATH} \
    --captioning_model_config_path ${CAPTIONING_CONFIG_PATH} \
    --server_url ${SERVER_URL} \
    --port ${PORT}
