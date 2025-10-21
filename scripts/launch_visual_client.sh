cd client

if ! [ -d "venv" ]; then
    python -m venv venv
fi

source venv/bin/activate

pip install -r requirements.txt

python -m src.demo.app_client_visual \
    --config_path ../configs/client/configs.yaml \
    --server_url http://localhost:8001 \
    --port 7862