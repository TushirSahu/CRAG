"""Typed, cached access to ``configs/config.yaml`` — the single source of truth.

Import ``get_settings()`` anywhere instead of hardcoding paths, ``k`` values, or
model names. Override the config location with the ``CRAG_CONFIG`` env var.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "config.yaml"


class VectorStoreCfg(BaseModel):
    provider: str = "lancedb"
    path: str = "./data/lancedb"
    table: str = "knowledge_base"


class EmbeddingsCfg(BaseModel):
    base_model: str = "all-MiniLM-L6-v2"
    finetuned_path: str = "./data/finetuned-domain-embeddings"


class RetrieverCfg(BaseModel):
    chunk_size: int = 500
    chunk_overlap: int = 50
    k_retrieval: int = 5
    fetch_k: int = 20
    hybrid: bool = True


class RaptorCfg(BaseModel):
    enabled: bool = True
    max_levels: int = 3
    min_cluster_size: int = 2
    reduce_dim: int = 10


class TemporalCfg(BaseModel):
    enabled: bool = True
    stale_after_days: int = 180


class TrustCfg(BaseModel):
    enabled: bool = True
    min_confidence: float = 0.5
    abstain_message: str = "I don't have enough grounded evidence to answer this confidently."
    unverified_prefix: str = "⚠️ I couldn't automatically verify this answer against the sources:\n\n"


class ModelsCfg(BaseModel):
    llm: str = "gemini-2.5-flash"
    grader_llm: str = "gemma-3-4b-it"
    generation_temperature: float = 0.0


class AgentCfg(BaseModel):
    max_retries: int = 2
    grade_rate_limit_seconds: float = 0.0


class CacheCfg(BaseModel):
    path: str = "./data/cache_db"
    threshold: float = 0.7


class SQLCfg(BaseModel):
    db_path: str = "./data/analytics.db"
    max_rows: int = 50


class Settings(BaseModel):
    vector_store: VectorStoreCfg = VectorStoreCfg()
    embeddings: EmbeddingsCfg = EmbeddingsCfg()
    retriever: RetrieverCfg = RetrieverCfg()
    raptor: RaptorCfg = RaptorCfg()
    temporal: TemporalCfg = TemporalCfg()
    trust: TrustCfg = TrustCfg()
    models: ModelsCfg = ModelsCfg()
    agent: AgentCfg = AgentCfg()
    cache: CacheCfg = CacheCfg()
    sql: SQLCfg = SQLCfg()

    def resolve(self, path: str) -> str:
        """Turn a config-relative path into an absolute one anchored at the repo root."""
        p = Path(path)
        return str(p if p.is_absolute() else (REPO_ROOT / p))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    config_path = Path(os.getenv("CRAG_CONFIG", DEFAULT_CONFIG_PATH))
    data = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    return Settings(**(data or {}))
