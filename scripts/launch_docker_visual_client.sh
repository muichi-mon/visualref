export CLIENT_CONFIG_PATH=configs/client/configs.yaml;
export PORT=7863;
docker compose --env-file .env.local build retrieval-client-visual
docker compose --env-file .env.local up retrieval-client-visual
