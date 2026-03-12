"""
config.py — Single source of truth for all settings.
Reads from environment variables / .env file via pydantic-settings.
"""
from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # ── LLM (Ollama local — free) ──────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.1:8b"          # swap to "mistral:7b" if preferred
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024

    # ── Embeddings (local sentence-transformers — free) ────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"   # 80MB, fast, good quality
    embedding_device: str = "cpu"                 # "cuda" if GPU available

    # ── ChromaDB (local, persistent — free) ───────────────────────────────
    chroma_persist_dir: str = "./data/chroma"
    collection_name: str = "pantheon_docs"

    # ── Chunking ───────────────────────────────────────────────────────────
    chunk_size: int = 800        # tokens approx.  Tune per doc type.
    chunk_overlap: int = 100     # overlap to preserve cross-chunk context

    # ── Retrieval ─────────────────────────────────────────────────────────
    top_k: int = 5               # chunks returned per query
    bm25_weight: float = 0.3     # weight for keyword search in hybrid mode
    semantic_weight: float = 0.7

    # ── Paths ─────────────────────────────────────────────────────────────
    docs_dir: str = "./data/docs"
    gold_set_path: str = "./data/gold_set.json"

    # ── Observability ─────────────────────────────────────────────────────
    mlflow_tracking_uri: str = "./data/mlflow"
    mlflow_experiment: str = "pantheon-rag"
    log_level: str = "INFO"
    log_file: str = "./data/rag.log"

    # ── Cost estimation (approx tokens/$ for Ollama = free local) ─────────
    cost_per_1k_input_tokens: float = 0.0    # local Ollama = $0
    cost_per_1k_output_tokens: float = 0.0


settings = Settings()
