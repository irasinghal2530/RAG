"""
Cohere reranker for RAG pipeline.

Uses Cohere's rerank API for high-quality reranking.

Usage:
    from src.reranking.cohere_reranker import CohereReranker
    
    reranker = CohereReranker(api_key="...")
    reranked = reranker.rerank(query, results, top_k=3)
"""

import os
from typing import List, Dict, Any, Optional

try:
    import cohere
    HAS_COHERE = True
except ImportError:
    HAS_COHERE = False


# Default Cohere rerank model
DEFAULT_MODEL = "rerank-english-v3.0"


class CohereReranker:
    """Reranker using Cohere's rerank API."""
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
    ):
        """
        Initialize the Cohere reranker.
        
        Args:
            model_name: Cohere rerank model name.
            api_key: Cohere API key (defaults to COHERE_API_KEY env var).
        """
        if not HAS_COHERE:
            raise ImportError(
                "cohere package not installed. "
                "Install with: pip install cohere"
            )
        
        self.model_name = model_name
        
        api_key = api_key or os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError(
                "Cohere API key required. "
                "Set COHERE_API_KEY environment variable or pass api_key."
            )
        
        self.client = cohere.Client(api_key)
    
    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        text_key: str = "document",
    ) -> List[Dict[str, Any]]:
        """
        Rerank results using Cohere API.
        
        Args:
            query: Search query.
            results: List of retrieval results.
            top_k: Number of results to return.
            text_key: Key for text content in results.
            
        Returns:
            list: Reranked results.
        """
        if not results:
            return []
        
        # Extract documents
        documents = [result.get(text_key, "") for result in results]
        
        # Call Cohere rerank API
        response = self.client.rerank(
            query=query,
            documents=documents,
            model=self.model_name,
            top_n=top_k or len(results),
        )
        
        # Build reranked results
        reranked = []
        for item in response.results:
            result = results[item.index].copy()
            result["rerank_score"] = item.relevance_score
            result["original_score"] = result.get("score", 0.0)
            result["original_index"] = item.index
            reranked.append(result)
        
        return reranked
    
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


if __name__ == "__main__":
    # Demo usage (requires COHERE_API_KEY)
    if os.getenv("COHERE_API_KEY"):
        sample_results = [
            {"chunk_id": "1", "document": "The maintenance schedule is every 1000 hours.", "score": 0.7},
            {"chunk_id": "2", "document": "Regular oil changes are important.", "score": 0.8},
            {"chunk_id": "3", "document": "Maintenance must be performed by certified technicians.", "score": 0.6},
        ]
        
        reranker = CohereReranker()
        reranked = reranker.rerank(
            "What is the maintenance schedule?",
            sample_results,
            top_k=2,
        )
        
        print("Reranked results:")
        for r in reranked:
            print(f"  {r['chunk_id']}: {r['rerank_score']:.4f}")
    else:
        print("Set COHERE_API_KEY to run this demo")
