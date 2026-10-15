"""
Evaluation module for RAG pipeline.

Provides metrics for retrieval quality, answer evaluation, and RAGAS assessment.

Modules:
    - retrieval_eval: Hit rate, MRR, NDCG, Precision metrics
    - answer_eval: Faithfulness and relevancy via LLM-as-judge
    - ragas_eval: RAGAS framework metrics (comprehensive RAG assessment)
"""

from evaluation.retrieval_eval import RetrievalEvaluator, evaluate_retrieval
from evaluation.answer_eval import AnswerEvaluator, evaluate_answer
from evaluation.ragas_eval import (
    RagasEvaluator,
    RagasInput,
    RagasMetrics,
    evaluate_ragas,
    create_ragas_dataset,
)

__all__ = [
    # Retrieval evaluation
    "RetrievalEvaluator",
    "evaluate_retrieval",
    # Answer evaluation
    "AnswerEvaluator",
    "evaluate_answer",
    # RAGAS evaluation
    "RagasEvaluator",
    "RagasInput",
    "RagasMetrics",
    "evaluate_ragas",
    "create_ragas_dataset",
]
