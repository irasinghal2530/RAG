"""
Embedding cache for RAG pipeline.

Caches computed embeddings to avoid redundant API calls
and speed up repeated queries.

Usage:
    from src.embedding.embedding_cache import EmbeddingCache
    
    cache = EmbeddingCache(cache_dir="data/cache")
    embedding = cache.get_or_compute("Hello world", embedder.embed_texts)
"""

import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable


class EmbeddingCache:
    """Caches embeddings to disk for efficiency."""
    
    def __init__(
        self,
        cache_dir: str | Path,
        max_cache_size_mb: int = 1000,
    ):
        """
        Initialize the embedding cache.
        
        Args:
            cache_dir: Directory to store cached embeddings.
            max_cache_size_mb: Maximum cache size in MB.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_size = max_cache_size_mb * 1024 * 1024
        
        # In-memory cache for frequently accessed embeddings
        self._memory_cache: Dict[str, List[float]] = {}
        self._memory_cache_size = 0
        self._max_memory_cache = 100 * 1024 * 1024  # 100MB
    
    def _hash_text(self, text: str) -> str:
        """Generate a hash for a text string."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def _get_cache_path(self, text_hash: str) -> Path:
        """Get the cache file path for a text hash."""
        # Use subdirectories to avoid too many files in one folder
        subdir = text_hash[:2]
        return self.cache_dir / subdir / f"{text_hash}.json"
    
    def get(self, text: str) -> Optional[List[float]]:
        """
        Get cached embedding for text.
        
        Args:
            text: Text to look up.
            
        Returns:
            Embedding vector if cached, None otherwise.
        """
        text_hash = self._hash_text(text)
        
        # Check memory cache first
        if text_hash in self._memory_cache:
            return self._memory_cache[text_hash]
        
        # Check disk cache
        cache_path = self._get_cache_path(text_hash)
        if cache_path.exists():
            try:
                data = json.loads(cache_path.read_text())
                embedding = data["embedding"]
                
                # Add to memory cache
                self._add_to_memory_cache(text_hash, embedding)
                
                return embedding
            except (json.JSONDecodeError, KeyError):
                return None
        
        return None
    
    def _add_to_memory_cache(
        self,
        text_hash: str,
        embedding: List[float],
    ):
        """Add embedding to memory cache with size limit."""
        size = len(embedding) * 8  # Approximate size in bytes
        
        # Evict if needed
        while self._memory_cache_size + size > self._max_memory_cache:
            if not self._memory_cache:
                break
            # Remove oldest entry (FIFO)
            oldest_key = next(iter(self._memory_cache))
            oldest_embedding = self._memory_cache.pop(oldest_key)
            self._memory_cache_size -= len(oldest_embedding) * 8
        
        self._memory_cache[text_hash] = embedding
        self._memory_cache_size += size
    
    def set(self, text: str, embedding: List[float]):
        """
        Cache an embedding for text.
        
        Args:
            text: Original text.
            embedding: Embedding vector.
        """
        text_hash = self._hash_text(text)
        
        # Save to disk
        cache_path = self._get_cache_path(text_hash)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        cache_path.write_text(
            json.dumps({
                "text_hash": text_hash,
                "embedding": embedding,
            })
        )
        
        # Add to memory cache
        self._add_to_memory_cache(text_hash, embedding)
    
    def get_or_compute(
        self,
        text: str,
        compute_fn: Callable[[List[str]], List[List[float]]],
    ) -> List[float]:
        """
        Get cached embedding or compute and cache it.
        
        Args:
            text: Text to embed.
            compute_fn: Function to compute embedding if not cached.
            
        Returns:
            Embedding vector.
        """
        embedding = self.get(text)
        
        if embedding is None:
            embeddings = compute_fn([text])
            embedding = embeddings[0]
            self.set(text, embedding)
        
        return embedding
    
    def batch_get_or_compute(
        self,
        texts: List[str],
        compute_fn: Callable[[List[str]], List[List[float]]],
    ) -> List[List[float]]:
        """
        Get cached embeddings or compute missing ones.
        
        Args:
            texts: List of texts to embed.
            compute_fn: Function to compute embeddings.
            
        Returns:
            List of embedding vectors.
        """
        results = [None] * len(texts)
        texts_to_compute = []
        indices_to_compute = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            embedding = self.get(text)
            if embedding is not None:
                results[i] = embedding
            else:
                texts_to_compute.append(text)
                indices_to_compute.append(i)
        
        # Compute missing embeddings
        if texts_to_compute:
            computed = compute_fn(texts_to_compute)
            
            for idx, text, embedding in zip(
                indices_to_compute,
                texts_to_compute,
                computed,
            ):
                results[idx] = embedding
                self.set(text, embedding)
        
        return results
    
    def clear(self):
        """Clear all cached embeddings."""
        import shutil
        
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._memory_cache.clear()
        self._memory_cache_size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_files = 0
        total_size = 0
        
        for cache_file in self.cache_dir.rglob("*.json"):
            total_files += 1
            total_size += cache_file.stat().st_size
        
        return {
            "cached_embeddings": total_files,
            "disk_size_mb": total_size / (1024 * 1024),
            "memory_cache_entries": len(self._memory_cache),
            "memory_cache_mb": self._memory_cache_size / (1024 * 1024),
        }


if __name__ == "__main__":
    # Demo usage
    cache = EmbeddingCache(
        cache_dir=Path(__file__).parent.parent.parent / "data" / "cache"
    )
    
    # Mock embedding function
    def mock_embed(texts):
        return [[0.1] * 384 for _ in texts]
    
    # Test caching
    text = "Hello world"
    
    embedding = cache.get_or_compute(text, mock_embed)
    print(f"First call: computed embedding with dim {len(embedding)}")
    
    embedding = cache.get_or_compute(text, mock_embed)
    print(f"Second call: retrieved from cache")
    
    stats = cache.get_stats()
    print(f"\nCache stats: {stats}")
