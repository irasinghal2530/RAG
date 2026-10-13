"""
Base reranker for RAG pipeline.

Uses cross-encoder models to rerank retrieval results
for improved relevance.

Usage:
    from src.reranking.base_reranker import BaseReranker
    
    reranker = BaseReranker()
    reranked = reranker.rerank(query, results, top_k=3)
"""

from typing import List, Dict, Any, Optional

try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False


# Default cross-encoder model
DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class BaseReranker:
    """Reranker using cross-encoder models."""
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
    ):
        """
        Initialize the reranker.
        
        Args:
            model_name: Cross-encoder model name.
        """
        if not HAS_CROSS_ENCODER:
            raise ImportError(
                "sentence-transformers required for reranking. "
                "Install with: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        self._model = None
    
    @property
    def model(self) -> "CrossEncoder":
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            print(f"Loading reranker model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            print("Reranker model loaded.")
        return self._model
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        text_key: str = "document",
    ) -> List[Dict[str, Any]]:
        """
        Rerank retrieval results using cross-encoder.
        
        Args:
            query: Search query.
            results: List of retrieval results.
            top_k: Number of results to return (None = all).
            text_key: Key for text content in results.
            
        Returns:
            list: Reranked results with updated scores.
        """
        if not results:
            return []
        
        # Create query-document pairs
        pairs = [
            [query, result.get(text_key, "")]
            for result in results
        ]
        
        # Get cross-encoder scores
        scores = self.model.predict(pairs)
        
        # Add scores to results
        scored_results = []
        for result, score in zip(results, scores):
            new_result = result.copy()
            new_result["rerank_score"] = float(score)
            new_result["original_score"] = result.get("score", 0.0)
            scored_results.append(new_result)
        
        # Sort by rerank score
        scored_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        
        # Limit results
        if top_k is not None:
            scored_results = scored_results[:top_k]
        
        return scored_results
    
    def batch_rerank(
        self,
        queries: List[str],
        results_list: List[List[Dict[str, Any]]],
        top_k: Optional[int] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Rerank multiple query-results pairs.
        
        Args:
            queries: List of queries.
            results_list: List of result lists.
            top_k: Number of results per query.
            
        Returns:
            list: List of reranked result lists.
        """
        return [
            self.rerank(query, results, top_k)
            for query, results in zip(queries, results_list)
        ]


def rerank(
    query: str,
    results: List[Dict[str, Any]],
    model_name: str = DEFAULT_MODEL,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function to rerank results.
    
    Args:
        query: Search query.
        results: Retrieval results.
        model_name: Reranker model name.
        top_k: Number of results to return.
        
    Returns:
        list: Reranked results.
    """
    reranker = BaseReranker(model_name=model_name)
    return reranker.rerank(query, results, top_k)


if __name__ == "__main__":
    # Demo usage
    sample_results = [
        {"chunk_id": "1", "document": "The maintenance schedule is every 1000 hours.", "score": 0.7},
        {"chunk_id": "2", "document": "Regular oil changes are important.", "score": 0.8},
        {"chunk_id": "3", "document": "Maintenance must be performed by certified technicians.", "score": 0.6},
    ]
    
    query = "What is the maintenance schedule?"
    
    if HAS_CROSS_ENCODER:
        reranker = BaseReranker()
        reranked = reranker.rerank(query, sample_results, top_k=2)
        
        print(f"Query: {query}\n")
        print("Reranked results:")
        for r in reranked:
            print(f"  {r['chunk_id']}: score={r['rerank_score']:.4f} (was {r['original_score']:.4f})")
            print(f"    {r['document'][:50]}...")
    else:
        print("Install sentence-transformers to run this demo")
