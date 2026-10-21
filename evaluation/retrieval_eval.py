"""
Retrieval evaluation for RAG pipeline.

Computes hit rate, MRR, NDCG and other retrieval metrics.

Usage:
    from evaluation.retrieval_eval import RetrievalEvaluator
    
    evaluator = RetrievalEvaluator()
    metrics = evaluator.evaluate(queries, ground_truth, retrieved)
"""

import math
from typing import List, Dict, Any, Set, Optional


class RetrievalEvaluator:
    """Evaluator for retrieval quality metrics."""
    
    def __init__(self, k_values: Optional[List[int]] = None):
        """
        Initialize the evaluator.

        Args:
            k_values: List of k values for metrics@k.
        """
        self.k_values = k_values or [1, 3, 5, 10]

    def _hit_rate(
        self,
        relevant: Set[str],
        retrieved: List[str],
        k: int,
    ) -> float:
        """Compute hit rate@k: 1 if any relevant id is in the top k, else 0."""
        return 1.0 if relevant & set(retrieved[:k]) else 0.0

    def _recall(
        self,
        relevant: Set[str],
        retrieved: List[str],
        k: int,
    ) -> float:
        """Compute recall@k: share of the relevant ids found in the top k."""
        if not relevant:
            return 0.0

        retrieved_at_k = set(retrieved[:k])
        hits = len(relevant & retrieved_at_k)
        return hits / len(relevant)
    
    def _precision(
        self,
        relevant: Set[str],
        retrieved: List[str],
        k: int,
    ) -> float:
        """Compute precision@k."""
        retrieved_at_k = retrieved[:k]
        if not retrieved_at_k:
            return 0.0
        
        hits = len(relevant & set(retrieved_at_k))
        return hits / len(retrieved_at_k)
    
    def _mrr(
        self,
        relevant: Set[str],
        retrieved: List[str],
    ) -> float:
        """Compute Mean Reciprocal Rank."""
        for i, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                return 1.0 / i
        return 0.0
    
    def _dcg(
        self,
        relevance_scores: List[float],
        k: int,
    ) -> float:
        """Compute Discounted Cumulative Gain."""
        dcg = 0.0
        for i, rel in enumerate(relevance_scores[:k]):
            dcg += rel / math.log2(i + 2)  # i+2 because i is 0-indexed
        return dcg
    
    def _ndcg(
        self,
        relevant: Set[str],
        retrieved: List[str],
        k: int,
    ) -> float:
        """Compute Normalized DCG."""
        # Actual relevance scores (binary)
        actual_rel = [1.0 if doc_id in relevant else 0.0 for doc_id in retrieved[:k]]
        
        # Ideal ranking puts every relevant id first, including ones that were
        # not retrieved; sorting actual_rel would score a 1-of-3 hit at rank 1 as 1.0.
        ideal_rel = [1.0] * min(len(relevant), k)
        
        dcg = self._dcg(actual_rel, k)
        idcg = self._dcg(ideal_rel, k)
        
        if idcg == 0:
            return 0.0
        return dcg / idcg
    
    def evaluate_single(
        self,
        relevant_ids: Set[str],
        retrieved_ids: List[str],
    ) -> Dict[str, float]:
        """
        Evaluate a single query.
        
        Args:
            relevant_ids: Set of relevant document IDs.
            retrieved_ids: List of retrieved document IDs (ranked).
            
        Returns:
            dict: Metrics for this query.
        """
        metrics = {}
        
        for k in self.k_values:
            metrics[f"hit_rate@{k}"] = self._hit_rate(relevant_ids, retrieved_ids, k)
            metrics[f"recall@{k}"] = self._recall(relevant_ids, retrieved_ids, k)
            metrics[f"precision@{k}"] = self._precision(relevant_ids, retrieved_ids, k)
            metrics[f"ndcg@{k}"] = self._ndcg(relevant_ids, retrieved_ids, k)
        
        metrics["mrr"] = self._mrr(relevant_ids, retrieved_ids)
        
        return metrics
    
    def evaluate(
        self,
        queries: List[Dict[str, Any]],
    ) -> Dict[str, float]:
        """
        Evaluate multiple queries and compute average metrics.
        
        Args:
            queries: List of dicts with 'relevant_ids' and 'retrieved_ids'.
            
        Returns:
            dict: Average metrics across all queries.
        """
        all_metrics = []
        
        for query in queries:
            relevant = set(query["relevant_ids"])
            retrieved = query["retrieved_ids"]
            metrics = self.evaluate_single(relevant, retrieved)
            all_metrics.append(metrics)
        
        # Compute averages
        avg_metrics = {}
        if all_metrics:
            for key in all_metrics[0].keys():
                avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        
        avg_metrics["num_queries"] = len(queries)
        return avg_metrics


def evaluate_retrieval(
    queries: List[Dict[str, Any]],
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Convenience function to evaluate retrieval.
    
    Args:
        queries: List of query evaluation data.
        k_values: List of k values for metrics.
        
    Returns:
        dict: Evaluation metrics.
    """
    evaluator = RetrievalEvaluator(k_values=k_values)
    return evaluator.evaluate(queries)


if __name__ == "__main__":
    # Demo usage
    sample_queries = [
        {
            "query": "What is the maintenance schedule?",
            "relevant_ids": {"doc1_chunk1", "doc1_chunk2"},
            "retrieved_ids": ["doc1_chunk1", "doc2_chunk5", "doc1_chunk2", "doc3_chunk1"],
        },
        {
            "query": "What are the pressure specs?",
            "relevant_ids": {"doc2_chunk3"},
            "retrieved_ids": ["doc1_chunk1", "doc2_chunk3", "doc3_chunk1"],
        },
    ]
    
    evaluator = RetrievalEvaluator()
    metrics = evaluator.evaluate(sample_queries)
    
    print("Retrieval Metrics:")
    for key, value in sorted(metrics.items()):
        print(f"  {key}: {value:.4f}")
