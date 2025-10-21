cd server

if ! [ -d "venv" ]; then
    python -m venv venv
fi

source venv/bin/activate

pip install -r requirements.txt

export CONFIG_PATH=configs/demo/clip_large.yaml;
export CAPTIONING_CONFIG_PATH=configs/captioning/llava_8bit.yaml;
export INDEX_PATH=faiss/coco/openai/clip-vit-large-patch14/image_index.faiss;
export LOGS_PATH=logs/retrieval_logs_clip_large.json;
python -m src.retrieval_server