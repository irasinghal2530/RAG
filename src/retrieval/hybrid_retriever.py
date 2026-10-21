"""
Hybrid (BM25 + dense) retrieval helpers for RAG pipeline.

Dense MiniLM vectors miss exact tokens such as model names ("ASD 40"),
pressures ("217 psig") and part names. BM25 catches those. The two ranked
lists are merged with Reciprocal Rank Fusion, which needs no score tuning.

Usage:
    from src.retrieval.hybrid_retriever import BM25Index, reciprocal_rank_fusion

    bm25 = BM25Index(ids, texts)
    sparse = bm25.search("ASD 40 capacity", top_k=20)
    fused = reciprocal_rank_fusion([dense_ids, [cid for cid, _ in sparse]])
"""

import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple


TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens; keeps numbers like '217' and '40'."""
    return TOKEN_PATTERN.findall(text.lower())


class BM25Index:
    """In-memory Okapi BM25 over the chunk corpus."""

    def __init__(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.ids = list(ids)
        self.k1 = k1
        self.b = b
        self.doc_tokens = [Counter(tokenize(t)) for t in texts]
        self.doc_lengths = [sum(c.values()) for c in self.doc_tokens]
        self.avg_length = (sum(self.doc_lengths) / len(self.doc_lengths)) if self.doc_lengths else 0.0

        doc_freq: Counter = Counter()
        for counts in self.doc_tokens:
            doc_freq.update(counts.keys())
        n = len(self.doc_tokens)
        self.idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5))
            for term, df in doc_freq.items()
        }

    def search(self, query: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """Return (chunk_id, score) pairs, best first, for chunks sharing a query term."""
        terms = [t for t in tokenize(query) if t in self.idf]
        if not terms:
            return []

        scores = []
        for chunk_id, counts, length in zip(self.ids, self.doc_tokens, self.doc_lengths):
            score = 0.0
            norm = self.k1 * (1 - self.b + self.b * length / (self.avg_length or 1))
            for term in terms:
                tf = counts.get(term, 0)
                if tf:
                    score += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
            if score > 0:
                scores.append((chunk_id, score))

        scores.sort(key=lambda pair: pair[1], reverse=True)
        return scores[:top_k]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
) -> List[Tuple[str, float]]:
    """Merge ranked id lists: score(id) = sum over lists of 1 / (k + rank)."""
    fused: Dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda pair: pair[1], reverse=True)
