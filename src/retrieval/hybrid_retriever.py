"""
Hybrid retriever for RAG pipeline.

Combines BM25 sparse retrieval with dense vector retrieval
for improved search quality.

Usage:
    from src.retrieval.hybrid_retriever import HybridRetriever
    
    retriever = HybridRetriever(vector_store_dir="data/vectordb")
    results = retriever.search("maintenance schedule", top_k=5)
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

from src.retrieval.retriever import VectorStore


class HybridRetriever:
    """Hybrid retriever combining BM25 and dense retrieval."""
    
    def __init__(
        self,
        vector_store_dir: str | Path,
        model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "documents",
        alpha: float = 0.5,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            vector_store_dir: Path to vector store.
            model_name: Embedding model name.
            collection_name: Collection name.
            alpha: Weight for dense retrieval (1-alpha for BM25).
            bm25_k1: BM25 k1 parameter.
            bm25_b: BM25 b parameter.
        """
        self.alpha = alpha
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        
        # Initialize vector store
        self.vector_store = VectorStore(
            persist_dir=vector_store_dir,
            collection_name=collection_name,
        )
        self.vector_store.get_or_create_collection()
        
        # Load embedding model
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Initialize BM25 index
        self._bm25 = None
        self._corpus = None
        self._chunk_ids = None
        
        if HAS_BM25:
            self._build_bm25_index()
        else:
            print("Warning: rank_bm25 not installed. Using dense-only retrieval.")
    
    def _build_bm25_index(self):
        """Build BM25 index from stored documents."""
        all_data = self.vector_store.collection.get(
            include=["documents", "metadatas"]
        )
        
        if not all_data["ids"]:
            print("No documents in collection for BM25 index")
            return
        
        self._chunk_ids = all_data["ids"]
        self._corpus = [
            self._tokenize(doc) for doc in all_data["documents"]
        ]
        self._bm25 = BM25Okapi(
            self._corpus,
            k1=self.bm25_k1,
            b=self.bm25_b,
        )
        
        print(f"BM25 index built with {len(self._corpus)} documents")
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization for BM25."""
        return text.lower().split()
    
    def _bm25_search(
        self,
        query: str,
        top_k: int = 100,
    ) -> Dict[str, float]:
        """
        Search using BM25.
        
        Args:
            query: Search query.
            top_k: Number of results.
            
        Returns:
            dict: Mapping of chunk_id to BM25 score.
        """
        if self._bm25 is None:
            return {}
        
        query_tokens = self._tokenize(query)
        scores = self._bm25.get_scores(query_tokens)
        
        # Create mapping of chunk_id to score
        results = {}
        for idx, score in enumerate(scores):
            if score > 0:
                results[self._chunk_ids[idx]] = score
        
        return results
    
    def _normalize_scores(
        self,
        scores: Dict[str, float],
    ) -> Dict[str, float]:
        """Normalize scores to 0-1 range."""
        if not scores:
            return {}
        
        max_score = max(scores.values())
        min_score = min(scores.values())
        
        if max_score == min_score:
            return {k: 1.0 for k in scores}
        
        return {
            k: (v - min_score) / (max_score - min_score)
            for k, v in scores.items()
        }
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        document_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search combining BM25 and dense retrieval.
        
        Args:
            query: Search query.
            top_k: Number of final results.
            document_id: Filter by document.
            
        Returns:
            list: Search results with combined scores.
        """
        # Build filter
        where = None
        if document_id:
            where = {"document_id": document_id}
        
        # Dense retrieval
        query_embedding = self.model.encode(query).tolist()
        dense_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k * 2,  # Get more for fusion
            where=where,
        )
        
        dense_scores = {
            r["chunk_id"]: r["score"] for r in dense_results
        }
        dense_scores = self._normalize_scores(dense_scores)
        
        # BM25 retrieval
        if HAS_BM25 and self._bm25 is not None:
            bm25_scores = self._bm25_search(query, top_k=top_k * 2)
            bm25_scores = self._normalize_scores(bm25_scores)
        else:
            bm25_scores = {}
        
        # Combine scores
        all_chunk_ids = set(dense_scores.keys()) | set(bm25_scores.keys())
        
        combined_scores = {}
        for chunk_id in all_chunk_ids:
            dense_score = dense_scores.get(chunk_id, 0.0)
            bm25_score = bm25_scores.get(chunk_id, 0.0)
            
            combined_scores[chunk_id] = (
                self.alpha * dense_score +
                (1 - self.alpha) * bm25_score
            )
        
        # Sort by combined score
        sorted_ids = sorted(
            combined_scores.keys(),
            key=lambda x: combined_scores[x],
            reverse=True,
        )[:top_k]
        
        # Build results
        results = []
        chunk_data = {r["chunk_id"]: r for r in dense_results}
        
        for chunk_id in sorted_ids:
            if chunk_id in chunk_data:
                result = chunk_data[chunk_id].copy()
                result["score"] = combined_scores[chunk_id]
                result["dense_score"] = dense_scores.get(chunk_id, 0.0)
                result["bm25_score"] = bm25_scores.get(chunk_id, 0.0)
                results.append(result)
        
        return results


if __name__ == "__main__":
    VECTOR_STORE_DIR = Path(__file__).parent.parent.parent / "data" / "vectordb"
    
    if VECTOR_STORE_DIR.exists():
        retriever = HybridRetriever(VECTOR_STORE_DIR, alpha=0.7)
        
        query = "What is the maintenance schedule?"
        results = retriever.search(query, top_k=3)
        
        print(f"Query: {query}\n")
        for r in results:
            print(f"Chunk: {r['chunk_id']}")
            print(f"  Combined: {r['score']:.4f}")
            print(f"  Dense: {r['dense_score']:.4f}")
            print(f"  BM25: {r['bm25_score']:.4f}")
            print()
    else:
        print(f"Directory not found: {VECTOR_STORE_DIR}")
