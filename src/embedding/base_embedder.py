"""
Base embedder for RAG pipeline.

Uses sentence-transformers for local embedding generation.

Usage:
    from src.embedding.base_embedder import BaseEmbedder
    
    embedder = BaseEmbedder()
    embeddings = embedder.embed_texts(["Hello world", "How are you?"])
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer


# Default model - good balance of speed and quality
DEFAULT_MODEL = "all-MiniLM-L6-v2"


class BaseEmbedder:
    """Base embedder using sentence-transformers."""
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        use_contextualized: bool = True,
    ):
        """
        Initialize the embedder.
        
        Args:
            model_name: Name of the sentence-transformer model.
            use_contextualized: Use contextualized text for embedding.
        """
        self.model_name = model_name
        self.use_contextualized = use_contextualized
        self._model = None
    
    @property
    def model(self) -> SentenceTransformer:
        """Lazy-load the embedding model."""
        if self._model is None:
            print(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            print(f"Model loaded. Dimension: {self._model.get_embedding_dimension()}")
        return self._model
    
    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.model.get_embedding_dimension()
    
    def embed_texts(
        self,
        texts: List[str],
        show_progress: bool = True,
    ) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed.
            show_progress: Show progress bar.
            
        Returns:
            list: List of embedding vectors.
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
        return embeddings.tolist()
    
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
        # Extract texts to embed
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
    
    def embed_document(
        self,
        chunks_path: Path,
        output_dir: Path,
    ) -> int:
        """
        Embed all chunks from a document file.
        
        Args:
            chunks_path: Path to chunks JSON file.
            output_dir: Directory to save embeddings.
            
        Returns:
            int: Number of chunks embedded.
        """
        # Load chunks
        data = json.loads(chunks_path.read_text(encoding="utf-8"))
        document_id = data["document_id"]
        chunks = data["chunks"]
        
        # Generate embeddings
        embedded_chunks = self.embed_chunks(chunks)
        
        # Prepare output data
        embedded_data = {
            "document_id": document_id,
            "total_chunks": len(embedded_chunks),
            "embedding_model": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "chunks": embedded_chunks,
        }
        
        # Save
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{document_id}.json"
        output_path.write_text(
            json.dumps(embedded_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        
        return len(embedded_chunks)


def embed_chunks(
    chunks: List[Dict[str, Any]],
    model_name: str = DEFAULT_MODEL,
    **kwargs,
) -> List[Dict[str, Any]]:
    """
    Convenience function to embed chunks.
    
    Args:
        chunks: List of chunk dictionaries.
        model_name: Embedding model name.
        **kwargs: Additional arguments for BaseEmbedder.
        
    Returns:
        list: Chunks with embeddings.
    """
    embedder = BaseEmbedder(model_name=model_name, **kwargs)
    return embedder.embed_chunks(chunks)


def batch_embed(
    chunks_dir: Path,
    output_dir: Path,
    model_name: str = DEFAULT_MODEL,
    **kwargs,
) -> int:
    """
    Embed all chunk files in a directory.
    
    Args:
        chunks_dir: Directory containing chunk JSON files.
        output_dir: Directory to save embedding files.
        model_name: Embedding model name.
        **kwargs: Additional arguments for BaseEmbedder.
        
    Returns:
        int: Total chunks embedded.
    """
    chunks_dir = Path(chunks_dir)
    output_dir = Path(output_dir)
    
    chunk_files = sorted(chunks_dir.glob("*.json"))
    
    if not chunk_files:
        print(f"No chunk files found in: {chunks_dir}")
        return 0
    
    print(f"Found {len(chunk_files)} chunk files to embed.")
    
    embedder = BaseEmbedder(model_name=model_name, **kwargs)
    total = 0
    
    for index, chunk_file in enumerate(chunk_files, start=1):
        print(f"[{index}/{len(chunk_files)}] Embedding: {chunk_file.name}")
        num = embedder.embed_document(chunk_file, output_dir)
        print(f"  -> {num} chunks embedded")
        total += num
    
    print("=" * 60)
    print(f"Embedding complete. Total: {total}")
    print("=" * 60)
    
    return total


if __name__ == "__main__":
    CHUNKS_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "chunks"
    EMBEDDINGS_DIR = Path(__file__).parent.parent.parent / "data" / "processed" / "embeddings"
    
    if CHUNKS_DIR.exists():
        batch_embed(CHUNKS_DIR, EMBEDDINGS_DIR)
    else:
        print(f"Directory not found: {CHUNKS_DIR}")
