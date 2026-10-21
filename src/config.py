"""
Config loader for RAG pipeline.

Reads config/retriever_config.yaml and config/llm_config.yaml so the values
in those files are the ones the pipeline actually runs with.

Usage:
    from src.config import load_retriever_config, load_llm_config

    cfg = load_retriever_config()
    top_k = cfg["retrieval"]["top_k"]
"""

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml


CONFIG_DIR = Path(__file__).parent.parent / "config"


@lru_cache(maxsize=None)
def _load_yaml(name: str) -> Dict[str, Any]:
    path = CONFIG_DIR / name
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_retriever_config() -> Dict[str, Any]:
    """Return the parsed retriever_config.yaml (empty dict if missing)."""
    return _load_yaml("retriever_config.yaml")


def load_llm_config() -> Dict[str, Any]:
    """Return the parsed llm_config.yaml (empty dict if missing)."""
    return _load_yaml("llm_config.yaml")


def load_document_titles() -> Dict[str, str]:
    """Return document_titles.yaml as {document_id: title}."""
    return _load_yaml("document_titles.yaml")
