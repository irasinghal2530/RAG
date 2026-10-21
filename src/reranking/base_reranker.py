"""
Cross-encoder reranker for RAG pipeline.

A bi-encoder (MiniLM) embeds query and chunk separately, so it only sees
topical similarity. A cross-encoder reads the query and chunk together and
scores whether the chunk answers the query, which puts the right chunk first.

Usage:
    from src.reranking import CrossEncoderReranker

    reranker = CrossEncoderReranker()
    top = reranker.rerank("What capacity does the ASD 40 have?", candidates, top_k=5)
"""

from typing import Any, Dict, List

from sentence_transformers import CrossEncoder


DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Reranks retrieval results with a sentence-transformers CrossEncoder."""

    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self._model = None

    @property
    def model(self) -> CrossEncoder:
        """Lazy-load the cross-encoder."""
        if self._model is None:
            print(f"Loading reranker model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Rescore results against the query and keep the best top_k.

        Args:
            query: Search query.
            results: Retrieval results with a 'document' field.
            top_k: Number of results to keep.

        Returns:
            list: Results sorted by 'rerank_score', best first.
        """
        if not results:
            return []

        pairs = [(query, r["document"]) for r in results]
        scores = self.model.predict(pairs, show_progress_bar=False)

        reranked = []
        for result, score in zip(results, scores):
            reranked.append({**result, "rerank_score": float(score)})

        reranked.sort(key=lambda r: r["rerank_score"], reverse=True)
        return reranked[:top_k]
