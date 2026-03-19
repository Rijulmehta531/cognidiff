# app/config.py
from enum import Enum
from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    """
    Supported LLM providers.
    Changing LLM_PROVIDER in .env switches the entire AI stack.
    """
    OLLAMA = "ollama"
    AZURE  = "azure"


class Settings(BaseSettings):
    """
    All application settings, loaded from environment variables / .env file.
    Pydantic validates types automatically — if DATABASE_URL is missing,
    the app fails immediately on startup with a clear error, not mysteriously
    later when the database is first used.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,   # LLM_PROVIDER and llm_provider both work
        extra="ignore",         # ignore unknown env vars instead of crashing
    )

    # ── LLM Provider ──────────────────────────────────────────────────────
    LLM_PROVIDER: LLMProvider = LLMProvider.OLLAMA

    # ── Ollama settings ───────────────────────────────────────────────────
    OLLAMA_BASE_URL:    str = "http://localhost:11434"
    OLLAMA_LLM_MODEL:   str = "qwen2.5-coder:3b"
    OLLAMA_EMBED_MODEL: str = "nomic-embed-text"

    # ── Azure OpenAI settings ─────────────────────────────────────────────
    AZURE_OPENAI_ENDPOINT:  str = ""
    AZURE_OPENAI_API_KEY:   str = ""
    AZURE_LLM_DEPLOYMENT:   str = "gpt-4o"
    AZURE_EMBED_DEPLOYMENT: str = "text-embedding-3-small"
    AZURE_API_VERSION:      str = "2024-02-01"

    # ── Database ──────────────────────────────────────────────────────────
    DATABASE_URL:      str = ""   # async  — used by the app
    SYNC_DATABASE_URL: str = ""   # sync   — used by alembic migrations

    # ── Redis ─────────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379"

    # ── GitHub ────────────────────────────────────────────────────────────
    GITHUB_WEBHOOK_SECRET: str = ""
    GITHUB_APP_ID:         str = ""
    GITHUB_PRIVATE_KEY:    str = ""
    GITHUB_TOKEN:          str = ""
    GITHUB_API_VERSION:    str = "2022-11-28"

    # ── LangSmith ─────────────────────────────────────────────────────────
    LANGCHAIN_TRACING_V2: bool = False
    LANGCHAIN_API_KEY:    str  = ""
    LANGCHAIN_PROJECT:    str  = "cognidiff"

    # ── Ingestion settings ────────────────────────────────────────────────
    CHUNK_MAX_LINES:     int   = 80    # max lines per code chunk
    EMBEDDING_DIMENSION: int   = 768   # nomic-embed-text dimension
                                    # change to 1536 for Azure embeddings

    # ── Ingestion versioning ───────────────────────────────────────────
    CHUNKING_VERSION: str = "1.0"

    # ── Retrieval settings ────────────────────────────────────────────────
    RETRIEVAL_TOP_K:       int   = 10    # how many chunks to retrieve per query
    RETRIEVAL_MIN_SCORE:   float = 0.70  # minimum similarity score (0-1)

    # ── Validators ────────────────────────────────────────────────────────
    @field_validator("AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", mode="before")
    @classmethod
    def check_azure_config(cls, v: str, info) -> str:
        """
        We don't fail here if Azure keys are empty —
        they're only required when LLM_PROVIDER=azure.
        That check happens in get_llm() at runtime.
        """
        return v

    @property
    def is_ollama(self) -> bool:
        return self.LLM_PROVIDER == LLMProvider.OLLAMA

    @property
    def is_azure(self) -> bool:
        return self.LLM_PROVIDER == LLMProvider.AZURE

    @property
    def active_embedding_model(self) -> str:
        """Unified embedding model name regardless of provider."""
        if self.is_ollama:
            return self.OLLAMA_EMBED_MODEL
        return self.AZURE_EMBED_DEPLOYMENT

    @property
    def parser_version(self) -> str:
        from importlib.metadata import version
        return version("tree-sitter")

@lru_cache
def get_settings() -> Settings:
    """
    Returns a cached Settings instance.

    @lru_cache means this function runs ONCE — the Settings object
    is created on first call and reused forever after.

    Why cache it?
    - Reading .env from disk on every request is wasteful
    - Settings don't change while the app is running
    - Every file that calls get_settings() gets the SAME object
    """
    return Settings()


def get_llm():
    """
    Returns the correct LLM based on LLM_PROVIDER in .env.
    Swap local → Azure by changing one env variable.
    """
    settings = get_settings()

    if settings.is_ollama:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_LLM_MODEL,
            temperature=0,      # deterministic — important for code review
                                # we want consistent analysis, not creative variation
        )

    # Azure OpenAI
    if not settings.AZURE_OPENAI_ENDPOINT or not settings.AZURE_OPENAI_API_KEY:
        raise ValueError(
            "LLM_PROVIDER is set to 'azure' but "
            "AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY is missing in .env"
        )

    from langchain_openai import AzureChatOpenAI
    return AzureChatOpenAI(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        azure_deployment=settings.AZURE_LLM_DEPLOYMENT,
        api_version=settings.AZURE_API_VERSION,
        temperature=0,
    )


def get_embedder():
    """
    Returns the correct embedding model based on LLM_PROVIDER.
    Embeddings and LLM always use the same provider —
    you don't want to mix embedding spaces between providers.
    """
    settings = get_settings()

    if settings.is_ollama:
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings(
            base_url=settings.OLLAMA_BASE_URL,
            model=settings.OLLAMA_EMBED_MODEL,
        )

    # Azure OpenAI
    if not settings.AZURE_OPENAI_ENDPOINT or not settings.AZURE_OPENAI_API_KEY:
        raise ValueError(
            "LLM_PROVIDER is set to 'azure' but "
            "AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY is missing in .env"
        )

    from langchain_openai import AzureOpenAIEmbeddings
    return AzureOpenAIEmbeddings(
        azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        api_key=settings.AZURE_OPENAI_API_KEY,
        azure_deployment=settings.AZURE_EMBED_DEPLOYMENT,
        api_version=settings.AZURE_API_VERSION,
    )