"""
OpenAI embedder for RAG pipeline.

Uses OpenAI's embedding API for high-quality embeddings.

Usage:
    from src.embedding.openai_embedder import OpenAIEmbedder
    
    embedder = OpenAIEmbedder(api_key="sk-...")
    embeddings = embedder.embed_texts(["Hello world"])
"""

import os
from typing import List, Dict, Any, Optional

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# Default OpenAI embedding model
DEFAULT_MODEL = "text-embedding-3-small"


class OpenAIEmbedder:
    """Embedder using OpenAI's embedding API."""
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        use_contextualized: bool = True,
    ):
        """
        Initialize the OpenAI embedder.
        
        Args:
            model_name: OpenAI embedding model name.
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var).
            use_contextualized: Use contextualized text for embedding.
        """
        if not HAS_OPENAI:
            raise ImportError(
                "openai package not installed. "
                "Install with: pip install openai"
            )
        
        self.model_name = model_name
        self.use_contextualized = use_contextualized
        
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required. "
                "Set OPENAI_API_KEY environment variable or pass api_key."
            )
        
        self.client = OpenAI(api_key=api_key)
        
        # Model dimensions
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
    
    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension for the model."""
        return self._dimensions.get(self.model_name, 1536)
    
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 100,
    ) -> List[List[float]]:
        """
        Generate embeddings using OpenAI API.
        
        Args:
            texts: List of texts to embed.
            batch_size: Number of texts per API call.
            
        Returns:
            list: List of embedding vectors.
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = self.client.embeddings.create(
                input=batch,
                model=self.model_name,
            )
            
            # Extract embeddings in order
            batch_embeddings = [
                item.embedding for item in response.data
            ]
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    def embed_chunks(
        self,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Generate embeddings for chunks.
        
        Args:
            chunks: List of chunk dictionaries.
            
        Returns:
            list: Chunks with embeddings added.
        """
        # Extract texts
        if self.use_contextualized:
            texts = [
                chunk.get("contextualized_text", chunk["text"])
                for chunk in chunks
            ]
        else:
            texts = [chunk["text"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = self.embed_texts(texts)
        
        # Add embeddings to chunks
        embedded_chunks = []
        for chunk, embedding in zip(chunks, embeddings):
            embedded_chunk = chunk.copy()
            embedded_chunk["embedding"] = embedding
            embedded_chunks.append(embedded_chunk)
        
        return embedded_chunks


if __name__ == "__main__":
    # Demo usage (requires OPENAI_API_KEY)
    import os
    
    if os.getenv("OPENAI_API_KEY"):
        embedder = OpenAIEmbedder()
        
        texts = ["Hello world", "How are you?"]
        embeddings = embedder.embed_texts(texts)
        
        print(f"Model: {embedder.model_name}")
        print(f"Dimension: {embedder.embedding_dimension}")
        print(f"Embedded {len(embeddings)} texts")
    else:
        print("Set OPENAI_API_KEY to run this demo")
