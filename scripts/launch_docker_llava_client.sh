export CLIENT_CONFIG_PATH=configs/client/configs.yaml
docker compose --env-file .env.local build retrieval-client
docker compose --env-file .env.local up retrieval-client
