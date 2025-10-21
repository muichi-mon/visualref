cd client

if ! [ -d "venv" ]; then
    python -m venv venv
fi

source venv/bin/activate

pip install -r requirements.txt

python -m src.demo.app_client \
    --config_path ../configs/client/configs.yaml \
    --captioning_model_config_path ../configs/captioning/llava_8bit.yaml \
    --server_url http://localhost:8000 \
    --port 7862