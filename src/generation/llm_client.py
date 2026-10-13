"""
LLM client for RAG pipeline.

Calls LLM APIs (OpenAI, Azure, etc.) to generate responses.

Usage:
    from src.generation.llm_client import LLMClient
    
    client = LLMClient()
    response = client.generate(messages)
"""

import os
from typing import List, Dict, Any, Optional

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# Default settings
DEFAULT_MODEL = "gpt-4"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 1024


class LLMClient:
    """Client for LLM API calls."""
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        """
        Initialize the LLM client.
        
        Args:
            model_name: LLM model name.
            api_key: API key (defaults to OPENAI_API_KEY env var).
            temperature: Sampling temperature.
            max_tokens: Maximum response tokens.
        """
        if not HAS_OPENAI:
            raise ImportError(
                "openai package not installed. "
                "Install with: pip install openai"
            )
        
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required. "
                "Set OPENAI_API_KEY environment variable or pass api_key."
            )
        
        self.client = OpenAI(api_key=api_key)
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generate a response from the LLM.
        
        Args:
            messages: Chat messages.
            temperature: Override temperature.
            max_tokens: Override max tokens.
            
        Returns:
            dict: Response with content and metadata.
        """
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
        )
        
        return {
            "content": response.choices[0].message.content,
            "model": response.model,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            "finish_reason": response.choices[0].finish_reason,
        }
    
    def generate_with_context(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate response using retrieved context.
        
        Args:
            query: User query.
            chunks: Retrieved chunks.
            system_prompt: Custom system prompt.
            
        Returns:
            dict: Response with content and sources.
        """
        from src.generation.prompt_builder import PromptBuilder
        
        builder = PromptBuilder(system_template=system_prompt)
        messages = builder.build_messages(query, chunks)
        
        response = self.generate(messages)
        
        # Add source information
        response["sources"] = [
            {
                "chunk_id": chunk.get("chunk_id"),
                "document_id": chunk.get("metadata", {}).get("document_id"),
                "score": chunk.get("score"),
            }
            for chunk in chunks
        ]
        
        return response
    
    def stream(
        self,
        messages: List[Dict[str, str]],
    ):
        """
        Stream response from the LLM.
        
        Args:
            messages: Chat messages.
            
        Yields:
            str: Response chunks.
        """
        stream = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


def generate_response(
    query: str,
    chunks: List[Dict[str, Any]],
    model_name: str = DEFAULT_MODEL,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to generate a RAG response.
    
    Args:
        query: User query.
        chunks: Retrieved chunks.
        model_name: LLM model name.
        **kwargs: Additional arguments for LLMClient.
        
    Returns:
        dict: Generated response.
    """
    client = LLMClient(model_name=model_name, **kwargs)
    return client.generate_with_context(query, chunks)


if __name__ == "__main__":
    # Demo usage (requires OPENAI_API_KEY)
    if os.getenv("OPENAI_API_KEY"):
        sample_chunks = [
            {
                "chunk_id": "MNT-001_chunk_001",
                "document": "The maintenance schedule requires checking oil levels every 1000 hours.",
                "metadata": {"document_id": "MNT-001"},
                "score": 0.85,
            },
        ]
        
        response = generate_response(
            "What is the maintenance schedule?",
            sample_chunks,
        )
        
        print(f"Response: {response['content']}")
        print(f"\nTokens used: {response['usage']['total_tokens']}")
        print(f"Sources: {response['sources']}")
    else:
        print("Set OPENAI_API_KEY to run this demo")
