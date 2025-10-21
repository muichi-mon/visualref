export CONFIG_PATH=configs/demo/clip_large.yaml;
export CAPTIONING_CONFIG_PATH=configs/captioning/llava_8bit.yaml;
export INDEX_PATH=faiss/movies/openai/clip-vit-large-patch14/image_index.faiss;
export LOGS_PATH=logs/retrieval_logs_clip_large.json;

docker compose build retrieval-server
docker compose up retrieval-server