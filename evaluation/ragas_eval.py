"""
RAGAS Evaluation for RAG Pipeline.

A self-contained implementation of the RAGAS metrics
(https://arxiv.org/abs/2309.15217); it does not use the `ragas` package.

The RAG generator answers questions. A separate, larger judge LLM scores
those answers. Every judge call runs in JSON mode so verdicts parse reliably.

When the judge's reply cannot be parsed, the metric is recorded as failed
(None) instead of 0.0, so a broken judge cannot pass for a broken pipeline.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


FAITHFULNESS_STATEMENT_EXTRACTION_PROMPT = """Break the answer into atomic factual claims that could be checked against a document.
Each claim must stand alone: replace pronouns with the names they refer to.
Skip filler such as "Here is the answer" and statements about missing information.
Return JSON only: {{"statements": ["claim 1", "claim 2"]}}

Question: {question}
Answer: {answer}

Example: {{"statements": ["The ASD 40 delivers 192 cfm at 125 psig."]}}"""


FAITHFULNESS_BATCH_VERIFY_PROMPT = """You verify whether statements are supported by the context.
A statement is supported if it can be directly inferred from the context, including by combining
facts from different passages. Paraphrase is allowed; numbers and units must match.
It is unsupported if it adds facts the context does not contain or contradicts it.
Return JSON only: {{"verdicts": [true, false]}} with exactly {n} booleans, one per statement, same order.

Context:
{context}

Statements:
{statements}
"""


ANSWER_RELEVANCY_PROMPT = """Write {n} different questions that the answer below is responding to.
Also decide if the answer is noncommittal: evasive, vague, or saying it cannot answer
(for example "I don't know" or "the context does not contain this").
Return JSON only: {{"questions": ["q1", "q2", "q3"], "noncommittal": false}}

Answer: {answer}
"""


CONTEXT_RELEVANCY_BATCH_PROMPT = """Rate how useful each context passage is for answering the question.
Score each passage from 0.0 to 1.0:
- 1.0 directly contains the facts needed to answer
- 0.5 related but missing key facts
- 0.0 unrelated
Return JSON only: {{"scores": [1.0, 0.0]}} with exactly {n} scores, one per passage, same order.

Question: {question}

Passages:
{passages}
"""


CONTEXT_PRECISION_BATCH_PROMPT = """For each context passage, decide if it was useful in arriving at the reference answer
to the question (it contains at least one fact the reference answer uses).
Return JSON only: {{"relevant": [true, false]}} with exactly {n} booleans, one per passage, same order.

Question: {question}

Reference answer: {reference}

Passages:
{passages}
"""


CONTEXT_RECALL_PROMPT = """Split the ground truth into atomic facts. For each fact, decide if the contexts support it.
Return JSON only, with one "supported" boolean per statement:
{{"statements": ["fact 1", "fact 2"], "supported": [true, false]}}

Ground Truth: {ground_truth}

Contexts:
{contexts}
"""


METRIC_NAMES = [
    "faithfulness",
    "answer_relevancy",
    "context_relevancy",
    "context_precision",
    "context_recall",
]


class JudgeOutputError(ValueError):
    """The judge reply could not be turned into a score."""


@dataclass
class RagasInput:
    """Input data structure for RAGAS evaluation."""
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RagasMetrics:
    """Output metrics from RAGAS evaluation. `failed` maps metric -> error."""
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    failed: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        return {k: getattr(self, k) for k in METRIC_NAMES if getattr(self, k) is not None}

    def __repr__(self) -> str:
        metrics = self.to_dict()
        text = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        if self.failed:
            text += f", failed={sorted(self.failed)}"
        return f"RagasMetrics({text})"


def _as_bool(value: Any) -> bool:
    """Strict truthiness: bool("false") is True in Python, so parse strings explicitly."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "supported", "relevant"}:
            return True
        if lowered in {"false", "no", "0", "not supported", "unsupported", "irrelevant"}:
            return False
    raise JudgeOutputError(f"not a boolean verdict: {value!r}")


class RagasEvaluator:
    """
    RAGAS evaluator. Uses a dedicated judge model, not the RAG generator.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
        max_context_chars: int = 16000,
    ):
        from src.generation.llm_client import create_judge_client

        self.client = None
        self.embedder = None
        self.max_context_chars = max_context_chars

        try:
            self.client = create_judge_client(model_name=model_name, api_key=api_key)
            self.model_name = self.client.model_name
        except (ImportError, ValueError):
            self.client = None
            self.model_name = model_name

        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(embedding_model or "all-MiniLM-L6-v2")
        except ImportError:
            pass

    def _call_judge(self, prompt: str) -> Any:
        """Call the judge in JSON mode and return the parsed object."""
        if not self.client:
            raise ValueError("Judge LLM not available. Set GROQ_API_KEY.")
        raw = self.client.generate_text(prompt, json_mode=True)
        parsed = self._extract_json(raw)
        if parsed is None:
            raise JudgeOutputError(f"judge returned no JSON: {raw[:200]!r}")
        return parsed

    def _extract_json(self, text: str) -> Any:
        cleaned = text.strip()
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        decoder = json.JSONDecoder()
        for match in re.finditer(r"[\[{]", cleaned):
            try:
                value, _ = decoder.raw_decode(cleaned[match.start():])
                return value
            except json.JSONDecodeError:
                continue
        return None

    def _get_list(self, parsed: Any, key: str, expected_len: Optional[int] = None) -> List[Any]:
        values = parsed.get(key) if isinstance(parsed, dict) else parsed
        if not isinstance(values, list):
            raise JudgeOutputError(f"expected a '{key}' list, got {str(parsed)[:200]}")
        if expected_len is not None and len(values) != expected_len:
            raise JudgeOutputError(
                f"'{key}' has {len(values)} items, expected {expected_len}"
            )
        return values

    def _format_passages(self, contexts: List[str]) -> str:
        parts = []
        for i, ctx in enumerate(contexts, start=1):
            clipped = ctx[:2500]
            parts.append(f"[{i}] {clipped}")
        return "\n\n".join(parts)

    def _join_contexts(self, contexts: List[str]) -> str:
        return "\n\n---\n\n".join(contexts)[: self.max_context_chars]

    def compute_faithfulness(
        self,
        question: str,
        answer: str,
        contexts: List[str],
    ) -> float:
        if not contexts or not answer:
            return 0.0

        extraction_prompt = FAITHFULNESS_STATEMENT_EXTRACTION_PROMPT.format(
            question=question,
            answer=answer,
        )
        statements = self._get_list(self._call_judge(extraction_prompt), "statements")
        statements = [str(s).strip() for s in statements if str(s).strip()]
        if not statements:
            # Nothing checkable (e.g. "I don't have enough information"): no hallucination.
            return 1.0

        numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(statements))
        verify_prompt = FAITHFULNESS_BATCH_VERIFY_PROMPT.format(
            context=self._join_contexts(contexts),
            statements=numbered,
            n=len(statements),
        )
        verdicts = self._get_list(self._call_judge(verify_prompt), "verdicts", len(statements))
        supported = sum(1 for v in verdicts if _as_bool(v))
        return supported / len(statements)

    def compute_answer_relevancy(
        self,
        question: str,
        answer: str,
        num_generations: int = 3,
    ) -> float:
        if not answer:
            return 0.0
        if not self.embedder:
            raise ValueError("sentence-transformers is required for answer_relevancy")

        prompt = ANSWER_RELEVANCY_PROMPT.format(answer=answer, n=num_generations)
        parsed = self._call_judge(prompt)
        generated = self._get_list(parsed, "questions")
        generated = [str(q).strip() for q in generated if str(q).strip()][:num_generations]
        if not generated:
            raise JudgeOutputError("judge generated no questions")

        if isinstance(parsed, dict) and _as_bool(parsed.get("noncommittal", False)):
            return 0.0

        vectors = self.embedder.encode([question] + generated, normalize_embeddings=True)
        similarities = [float(vectors[0] @ vec) for vec in vectors[1:]]
        return sum(similarities) / len(similarities)

    def compute_context_relevancy(
        self,
        question: str,
        contexts: List[str],
    ) -> float:
        if not contexts:
            return 0.0

        prompt = CONTEXT_RELEVANCY_BATCH_PROMPT.format(
            question=question,
            passages=self._format_passages(contexts),
            n=len(contexts),
        )
        scores = self._get_list(self._call_judge(prompt), "scores", len(contexts))
        numeric = []
        for score in scores:
            try:
                numeric.append(max(0.0, min(1.0, float(score))))
            except (TypeError, ValueError):
                raise JudgeOutputError(f"not a score: {score!r}")
        return sum(numeric) / len(numeric)

    def compute_context_precision(
        self,
        question: str,
        contexts: List[str],
        reference: Optional[str] = None,
    ) -> float:
        """
        Average precision of the ranking. Relevance is judged against the
        reference answer (ground truth) when given, as RAGAS does; otherwise
        against the generated answer, or the question as a last resort.
        """
        if not contexts:
            return 0.0

        prompt = CONTEXT_PRECISION_BATCH_PROMPT.format(
            question=question,
            reference=reference or "(not available; judge usefulness for the question)",
            passages=self._format_passages(contexts),
            n=len(contexts),
        )
        flags = self._get_list(self._call_judge(prompt), "relevant", len(contexts))
        relevancy = [_as_bool(v) for v in flags]
        if not any(relevancy):
            return 0.0

        precisions = []
        relevant_count = 0
        for i, is_rel in enumerate(relevancy):
            if is_rel:
                relevant_count += 1
                precisions.append(relevant_count / (i + 1))
        return sum(precisions) / len(precisions)

    def compute_context_recall(
        self,
        contexts: List[str],
        ground_truth: str,
    ) -> float:
        if not contexts or not ground_truth:
            return 0.0

        prompt = CONTEXT_RECALL_PROMPT.format(
            ground_truth=ground_truth,
            contexts=self._join_contexts(contexts),
        )
        parsed = self._call_judge(prompt)
        statements = self._get_list(parsed, "statements")
        supported = self._get_list(parsed, "supported", len(statements))
        if not supported:
            raise JudgeOutputError("judge returned no ground-truth statements")
        return sum(1 for s in supported if _as_bool(s)) / len(supported)

    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        metrics: Optional[List[str]] = None,
    ) -> RagasMetrics:
        available_metrics = {
            "faithfulness": lambda: self.compute_faithfulness(question, answer, contexts),
            "answer_relevancy": lambda: self.compute_answer_relevancy(question, answer),
            "context_relevancy": lambda: self.compute_context_relevancy(question, contexts),
            "context_precision": lambda: self.compute_context_precision(
                question, contexts, reference=ground_truth or answer or None
            ),
        }
        if ground_truth:
            available_metrics["context_recall"] = lambda: self.compute_context_recall(
                contexts, ground_truth
            )

        if metrics is None:
            metrics_to_compute = list(available_metrics.keys())
        else:
            metrics_to_compute = [m for m in metrics if m in available_metrics]

        results = RagasMetrics()
        for metric_name in metrics_to_compute:
            try:
                score = available_metrics[metric_name]()
                setattr(results, metric_name, score)
            except Exception as e:
                results.failed[metric_name] = f"{type(e).__name__}: {e}"
                print(f"Warning: Failed to compute {metric_name}: {e}")
        return results

    def evaluate_batch(
        self,
        samples: List[RagasInput],
        metrics: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        all_results = []
        all_failures = []
        for i, sample in enumerate(samples):
            if verbose:
                print(f"Evaluating sample {i+1}/{len(samples)}...")
            result = self.evaluate(
                question=sample.question,
                answer=sample.answer,
                contexts=sample.contexts,
                ground_truth=sample.ground_truth,
                metrics=metrics,
            )
            all_results.append(result.to_dict())
            all_failures.append(result.failed)

        aggregates = {}
        failure_counts = {}
        for key in METRIC_NAMES:
            values = [r[key] for r in all_results if key in r]
            if values:
                aggregates[f"avg_{key}"] = sum(values) / len(values)
            failed = sum(1 for f in all_failures if key in f)
            if failed:
                failure_counts[key] = failed

        return {
            "aggregate_metrics": aggregates,
            "failure_counts": failure_counts,
            "per_sample_metrics": all_results,
            "per_sample_failures": all_failures,
            "num_samples": len(samples),
        }


def evaluate_ragas(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None,
    **kwargs,
) -> Dict[str, float]:
    evaluator = RagasEvaluator(**kwargs)
    result = evaluator.evaluate(question, answer, contexts, ground_truth)
    return result.to_dict()


def create_ragas_dataset(
    eval_data: List[Dict[str, Any]],
) -> List[RagasInput]:
    samples = []
    for item in eval_data:
        sample = RagasInput(
            question=item.get("question", ""),
            answer=item.get("answer", ""),
            contexts=item.get("contexts", item.get("retrieved_chunks", [])),
            ground_truth=item.get("ground_truth", item.get("expected_answer")),
        )
        samples.append(sample)
    return samples


if __name__ == "__main__":
    sample = RagasInput(
        question="What is the maintenance schedule?",
        answer="The maintenance schedule requires checking oil levels every 1000 hours and replacing filters during regular intervals.",
        contexts=[
            "The maintenance schedule requires checking oil levels every 1000 hours.",
            "All filters should be replaced during regular maintenance intervals.",
            "Equipment must be inspected before each use for safety compliance.",
        ],
        ground_truth="Oil levels must be checked every 1000 hours. Filters need regular replacement.",
    )

    evaluator = RagasEvaluator()
    if evaluator.client:
        print("Computing RAGAS metrics...")
        print("=" * 60)
        print(f"Judge model: {evaluator.model_name}")
        print(f"Question: {sample.question}")
        print(f"Answer: {sample.answer}")
        print("=" * 60)
        metrics = evaluator.evaluate(
            question=sample.question,
            answer=sample.answer,
            contexts=sample.contexts,
            ground_truth=sample.ground_truth,
        )
        print("\nRAGAS Metrics:")
        for name, value in metrics.to_dict().items():
            print(f"  {name}: {value:.4f}")
        for name, error in metrics.failed.items():
            print(f"  {name}: FAILED ({error})")
    else:
        print("Set GROQ_API_KEY for RAGAS evaluation")
