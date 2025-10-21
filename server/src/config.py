# server/src/config.py
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    base_dir: Path = Path(".").resolve()

    config_path: Path 
    index_path: Path
    logs_path: Path
    captioning_config_path: Path | None = None

    @field_validator(
        "config_path",
        "captioning_config_path",
        "index_path",
        "logs_path",
        mode="before",
    )
    @classmethod
    def _resolve_paths(cls, v, info):
        if v is None:
            return v
        p = Path(v)
        base: Path = info.data.get("base_dir", Path(".").resolve())
        return p if p.is_absolute() else (base / p)


settings = ServerSettings()


def resolve_repo(p: str | Path) -> str:
    p = Path(p)
    base = settings.base_dir
    return str(p if p.is_absolute() else (base / p))
