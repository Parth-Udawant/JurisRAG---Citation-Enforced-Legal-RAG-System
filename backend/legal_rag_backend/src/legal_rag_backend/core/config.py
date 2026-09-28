from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

CONFIG_DIR = Path(__file__).resolve().parent

ENV_FILE = None
for candidate in [CONFIG_DIR, *CONFIG_DIR.parents]:
    env_candidate = candidate / ".env"
    if env_candidate.exists():
        ENV_FILE = env_candidate
        load_dotenv(env_candidate)
        break



def required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@dataclass(frozen=True)
class Settings:
    azure_openai_endpoint: str
    azure_openai_api_key: str
    azure_openai_api_version: str
    embedding_deployment: str
    embedding_dimensions: int
    gpt_deployment: str
    search_endpoint: str
    search_admin_key: str
    search_index_name: str

    candidate_k: int = 10
    final_k: int = 5
    vector_filter_mode: str = "preFilter"
    gpt_max_completion_tokens: int = 1800

    embedding_usd_per_1m_tokens: float = 0.0
    gpt_input_usd_per_1m_tokens: float = 0.0
    gpt_output_usd_per_1m_tokens: float = 0.0
    search_usd_per_operation: float = 0.0

    cors_origins: tuple[str, ...] = ("http://localhost:3000", "http://localhost:5173")
    api_key: str | None = None


def get_settings() -> Settings:
    origins = tuple(
        x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",") if x.strip()
    )
    return Settings(
        azure_openai_endpoint=required("AZURE_OPENAI_ENDPOINT"),
        azure_openai_api_key=required("AZURE_OPENAI_API_KEY"),
        azure_openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        embedding_deployment=required("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        embedding_dimensions=int(os.getenv("AZURE_OPENAI_EMBEDDING_DIMENSIONS", "1536")),
        gpt_deployment=required("AZURE_OPENAI_GPT_DEPLOYMENT"),
        search_endpoint=required("AZURE_SEARCH_ENDPOINT"),
        search_admin_key=required("AZURE_SEARCH_ADMIN_KEY"),
        search_index_name=os.getenv("AZURE_SEARCH_INDEX_NAME", "legal-rag-search-vectors"),
        candidate_k=int(os.getenv("RAG_CANDIDATE_K", "10")),
        final_k=int(os.getenv("RAG_FINAL_K", "5")),
        vector_filter_mode=os.getenv("AZURE_SEARCH_VECTOR_FILTER_MODE", "preFilter"),
        gpt_max_completion_tokens=int(os.getenv("GPT_MAX_TOKENS", "1800")),
        embedding_usd_per_1m_tokens=float(os.getenv("EMBEDDING_USD_PER_1M_TOKENS", "0")),
        gpt_input_usd_per_1m_tokens=float(os.getenv("GPT_INPUT_USD_PER_1M_TOKENS", "0")),
        gpt_output_usd_per_1m_tokens=float(os.getenv("GPT_OUTPUT_USD_PER_1M_TOKENS", "0")),
        search_usd_per_operation=float(os.getenv("SEARCH_USD_PER_OPERATION", "0")),
        cors_origins=origins,
        api_key=os.getenv("BACKEND_API_KEY") or None,
    )
