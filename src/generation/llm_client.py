"""
LLM client for RAG pipeline.

Calls Google Gemini to generate responses.

Usage:
    from src.generation.llm_client import LLMClient

    client = LLMClient()
    response = client.generate(messages)
"""

import os
from typing import List, Dict, Any, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from google import genai
    from google.genai import types
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024


def resolve_google_api_key(api_key: Optional[str] = None) -> Optional[str]:
    """Return a real Gemini key, ignoring empty/.env placeholders."""
    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        return None
    key = key.strip().strip('"').strip("'")
    if not key or key.lower().startswith("your-"):
        return None
    return key


def resolve_model_name(model_name: Optional[str] = None) -> str:
    """Return the Gemini model id from argument, LLM_MODEL, or the default."""
    name = model_name or os.getenv("LLM_MODEL") or DEFAULT_MODEL
    return name.strip().strip('"').strip("'")


class LLMClient:
    """Client for Google Gemini API calls."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        if not HAS_GEMINI:
            raise ImportError(
                "google-genai package not installed. "
                "Install with: pip install google-genai"
            )

        self.model_name = resolve_model_name(model_name)
        self.temperature = temperature
        self.max_tokens = max_tokens

        api_key = resolve_google_api_key(api_key)
        if not api_key:
            raise ValueError(
                "Google API key required. "
                "Set GOOGLE_API_KEY in .env (https://aistudio.google.com/apikey)."
            )

        self.client = genai.Client(api_key=api_key)

    def _split_messages(self, messages: List[Dict[str, str]]) -> tuple:
        """Split chat messages into Gemini system instruction and contents."""
        system_prompt = None
        parts = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_prompt = content
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(content)

        return system_prompt, "\n\n".join(parts)

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Generate a response from Gemini."""
        system_prompt, contents = self._split_messages(messages)

        config = types.GenerateContentConfig(
            temperature=self.temperature if temperature is None else temperature,
            max_output_tokens=max_tokens or self.max_tokens,
            system_instruction=system_prompt,
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )

        usage = {}
        if getattr(response, "usage_metadata", None):
            usage = {
                "prompt_tokens": getattr(response.usage_metadata, "prompt_token_count", 0) or 0,
                "completion_tokens": getattr(response.usage_metadata, "candidates_token_count", 0) or 0,
                "total_tokens": getattr(response.usage_metadata, "total_token_count", 0) or 0,
            }

        return {
            "content": response.text or "",
            "model": self.model_name,
            "usage": usage,
            "finish_reason": "stop",
        }

    def generate_text(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Generate plain text from a single prompt."""
        response = self.generate(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response["content"].strip()

    def generate_with_context(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate a response using retrieved context."""
        from src.generation.prompt_builder import PromptBuilder

        builder = PromptBuilder(system_template=system_prompt)
        messages = builder.build_messages(query, chunks)
        response = self.generate(messages)
        response["sources"] = [
            {
                "chunk_id": chunk.get("chunk_id"),
                "document_id": chunk.get("metadata", {}).get("document_id"),
                "score": chunk.get("score"),
            }
            for chunk in chunks
        ]
        return response

    def stream(self, messages: List[Dict[str, str]]):
        """Stream response chunks from Gemini."""
        system_prompt, contents = self._split_messages(messages)
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            system_instruction=system_prompt,
        )

        stream = self.client.models.generate_content_stream(
            model=self.model_name,
            contents=contents,
            config=config,
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text


def generate_response(
    query: str,
    chunks: List[Dict[str, Any]],
    model_name: Optional[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Convenience function to generate a RAG response."""
    client = LLMClient(model_name=model_name, **kwargs)
    return client.generate_with_context(query, chunks)


if __name__ == "__main__":
    if resolve_google_api_key():
        sample_chunks = [
            {
                "chunk_id": "MNT-001_chunk_001",
                "document": "The maintenance schedule requires checking oil levels every 1000 hours.",
                "metadata": {"document_id": "MNT-001"},
                "score": 0.85,
            },
        ]
        response = generate_response("What is the maintenance schedule?", sample_chunks)
        print(f"Response: {response['content']}")
        print(f"\nTokens used: {response['usage'].get('total_tokens', 'N/A')}")
        print(f"Sources: {response['sources']}")
    else:
        print("Set GOOGLE_API_KEY to run this demo")
