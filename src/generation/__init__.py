"""
Generation module for RAG pipeline.

Handles prompt building and LLM response generation.
"""

from src.generation.prompt_builder import PromptBuilder, build_prompt
from src.generation.llm_client import (
    LLMClient,
    create_judge_client,
    generate_response,
    resolve_groq_api_key,
    resolve_judge_model_name,
)

__all__ = [
    "PromptBuilder",
    "build_prompt",
    "LLMClient",
    "create_judge_client",
    "generate_response",
    "resolve_groq_api_key",
    "resolve_judge_model_name",
]
