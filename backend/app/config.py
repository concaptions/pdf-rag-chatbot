from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    # Default Claude model; must be one of llm.MODELS["anthropic"], otherwise the first entry is used.
    anthropic_model: str = "claude-sonnet-5-5"
    openai_api_key: str = ""
    openai_model: str = "gpt-4.1"
    # Provider tried first when the user has not picked one; the other one is the automatic fallback.
    llm_primary: str = "anthropic"
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 2

    # Conversation memory: the model always sees at least `history_window` recent messages verbatim.
    # When `history_compact_at` messages are unsummarised, older ones are folded into a running summary.
    history_window: int = 30
    history_compact_at: int = 50

    supabase_url: str = ""
    supabase_service_role_key: str = ""

    embedding_model: str = "BAAI/bge-small-en-v1.5"
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 8
    # Passages below this cosine similarity are never sent to the model (pure noise for bge-small).
    min_similarity: float = 0.3

    max_upload_mb: int = 25
    max_files_per_request: int = 10
    # A document with no indexing progress for this long is marked failed (worker restarted mid-job).
    stale_processing_s: int = 180
    allowed_origins: str = "http://localhost:5173"

    @property
    def origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
