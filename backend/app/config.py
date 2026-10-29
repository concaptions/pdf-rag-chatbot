from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-4-8"
    anthropic_timeout_s: float = 60.0
    anthropic_max_retries: int = 2

    openai_api_key: str = ""
    openai_model: str = "gpt-4.1"
    # Provider tried first; the other one is the automatic fallback.
    llm_primary: str = "anthropic"

    supabase_url: str = ""
    supabase_service_role_key: str = ""

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 6
    min_similarity: float = 0.45

    max_upload_mb: int = 25
    max_files_per_request: int = 10
    allowed_origins: str = "http://localhost:5173"

    @property
    def origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
