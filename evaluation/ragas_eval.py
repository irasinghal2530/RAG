"""
RAGAS Evaluation for RAG Pipeline.

Implements RAGAS (Retrieval Augmented Generation Assessment) metrics:
- Faithfulness: How grounded is the answer in the context
- Answer Relevancy: How relevant is the answer to the question
- Context Relevancy: How relevant is the retrieved context to the question
- Context Precision: Are relevant chunks ranked higher
- Context Recall: Does the context cover the ground truth answer

Usage:
    from evaluation.ragas_eval import RagasEvaluator
    
    evaluator = RagasEvaluator()
    metrics = evaluator.evaluate(
        question="What is the maintenance schedule?",
        answer="Oil should be checked every 1000 hours.",
        contexts=["Oil must be checked every 1000 hours...", ...],
        ground_truth="The maintenance requires oil checks every 1000 hours."
    )

Reference:
    RAGAS Paper: https://arxiv.org/abs/2309.15217
"""

import re
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ==============================================================================
# Prompt Templates for RAGAS Metrics
# ==============================================================================

FAITHFULNESS_STATEMENT_EXTRACTION_PROMPT = """Given a question and answer, extract all factual statements from the answer.
Each statement should be a simple, atomic claim that can be verified independently.

Question: {question}
Answer: {answer}

Extract all statements as a JSON list of strings. Only output the JSON list, nothing else.
Example output: ["Statement 1", "Statement 2", "Statement 3"]

Statements:"""


FAITHFULNESS_VERIFICATION_PROMPT = """Given a context and a statement, determine if the statement is supported by the context.

Context:
{context}

Statement: {statement}

Can this statement be inferred from the context? 
Respond with exactly "Yes" or "No"."""


ANSWER_RELEVANCY_PROMPT = """Given the following answer, generate a question that this answer is trying to address.
Be specific and try to capture the main intent of the answer.

Answer: {answer}

Generated question:"""


CONTEXT_RELEVANCY_PROMPT = """Given a question and a context passage, rate how relevant the context is to answering the question.

Question: {question}

Context: {context}

Rate the relevance on a scale from 0 to 1:
- 1.0: Directly relevant, contains information needed to answer the question
- 0.5: Partially relevant, contains some related information
- 0.0: Not relevant, no useful information for the question

Respond with just a number between 0 and 1."""


CONTEXT_RECALL_PROMPT = """Given a ground truth answer and a set of context passages, determine what fraction of the ground truth information is present in the contexts.

Ground Truth: {ground_truth}

Contexts:
{contexts}

Extract statements from the ground truth and check if each is supported by the contexts.
Respond with a JSON object containing:
- "statements": list of statements from ground truth
- "supported": list of booleans indicating if each statement is supported

Example:
{{"statements": ["fact 1", "fact 2"], "supported": [true, false]}}

Response:"""


CONTEXT_PRECISION_PROMPT = """Given a question and a context passage, determine if this context is useful for answering the question.

Question: {question}
Context: {context}

Is this context useful for answering the question? Respond with exactly "Yes" or "No"."""


# ==============================================================================
# Data Classes
# ==============================================================================

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
    """Output metrics from RAGAS evaluation."""
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def __repr__(self) -> str:
        metrics = self.to_dict()
        return "RagasMetrics(" + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()) + ")"


# ==============================================================================
# RAGAS Evaluator
# ==============================================================================

class RagasEvaluator:
    """
    RAGAS Evaluator for RAG pipeline assessment.
    
    Implements the following metrics:
    
    1. Faithfulness (0-1): Measures if the answer is grounded in the context.
       Higher = answer claims are supported by context.
       
    2. Answer Relevancy (0-1): Measures if the answer addresses the question.
       Uses reverse question generation and cosine similarity.
       
    3. Context Relevancy (0-1): Measures if retrieved contexts are relevant.
       Average relevancy score across all context chunks.
       
    4. Context Precision (0-1): Measures if relevant chunks appear first.
       Weighted by rank position (earlier = better).
       
    5. Context Recall (0-1): Measures if context covers ground truth.
       Fraction of ground truth statements found in context.
    """
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """
        Initialize the RAGAS evaluator.
        
        Args:
            model_name: Gemini model for evaluation.
            api_key: Google API key (defaults to GOOGLE_API_KEY env var).
            embedding_model: Model for embeddings (for answer relevancy).
        """
        from src.generation.llm_client import LLMClient

        self.client = None
        self.embedder = None

        try:
            self.client = LLMClient(model_name=model_name, api_key=api_key)
            self.model_name = self.client.model_name
        except (ImportError, ValueError):
            self.client = None
            self.model_name = model_name
        
        # Try to load sentence transformer for embeddings
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(embedding_model or "all-MiniLM-L6-v2")
        except ImportError:
            pass
    
    def _call_llm(self, prompt: str, max_tokens: int = 500) -> str:
        """Call LLM and return response text."""
        if not self.client:
            raise ValueError("Gemini client not available. Set GOOGLE_API_KEY.")
        return self.client.generate_text(prompt, max_tokens=max_tokens)
    
    def _extract_json(self, text: str) -> Any:
        """Extract JSON from LLM response."""
        # Try direct parsing
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try to find JSON in the text
        json_match = re.search(r'\[.*?\]|\{.*?\}', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors."""
        import math
        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)
    
    # --------------------------------------------------------------------------
    # Faithfulness Metric
    # --------------------------------------------------------------------------
    
    def compute_faithfulness(
        self,
        question: str,
        answer: str,
        contexts: List[str],
    ) -> float:
        """
        Compute faithfulness score.
        
        Measures whether the claims in the answer are supported by the context.
        
        Process:
        1. Extract factual statements from the answer
        2. For each statement, verify if it can be inferred from context
        3. Faithfulness = supported_statements / total_statements
        
        Args:
            question: The user question.
            answer: The generated answer.
            contexts: List of retrieved context strings.
            
        Returns:
            float: Faithfulness score (0-1).
        """
        if not contexts or not answer:
            return 0.0
        
        # Step 1: Extract statements from answer
        extraction_prompt = FAITHFULNESS_STATEMENT_EXTRACTION_PROMPT.format(
            question=question,
            answer=answer,
        )
        
        response = self._call_llm(extraction_prompt)
        statements = self._extract_json(response)
        
        if not statements or not isinstance(statements, list):
            # Fallback: treat entire answer as one statement
            statements = [answer]
        
        # Step 2: Verify each statement against context
        combined_context = "\n\n".join(contexts)
        supported_count = 0
        
        for statement in statements:
            verification_prompt = FAITHFULNESS_VERIFICATION_PROMPT.format(
                context=combined_context,
                statement=statement,
            )
            
            verdict = self._call_llm(verification_prompt, max_tokens=10)
            if verdict.lower().strip() in ["yes", "true", "1"]:
                supported_count += 1
        
        return supported_count / len(statements) if statements else 0.0
    
    # --------------------------------------------------------------------------
    # Answer Relevancy Metric
    # --------------------------------------------------------------------------
    
    def compute_answer_relevancy(
        self,
        question: str,
        answer: str,
        num_generations: int = 3,
    ) -> float:
        """
        Compute answer relevancy score.
        
        Measures whether the answer addresses the question using reverse
        question generation and semantic similarity.
        
        Process:
        1. Generate N questions that the answer might be addressing
        2. Compute embedding similarity between original and generated questions
        3. Answer Relevancy = average similarity
        
        Args:
            question: The original question.
            answer: The generated answer.
            num_generations: Number of questions to generate.
            
        Returns:
            float: Answer relevancy score (0-1).
        """
        if not answer or not self.embedder:
            return 0.0
        
        # Generate reverse questions
        generated_questions = []
        for _ in range(num_generations):
            prompt = ANSWER_RELEVANCY_PROMPT.format(answer=answer)
            gen_question = self._call_llm(prompt, max_tokens=100)
            generated_questions.append(gen_question)
        
        # Compute embeddings
        original_embedding = self.embedder.encode(question).tolist()
        
        similarities = []
        for gen_q in generated_questions:
            gen_embedding = self.embedder.encode(gen_q).tolist()
            sim = self._cosine_similarity(original_embedding, gen_embedding)
            similarities.append(sim)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    # --------------------------------------------------------------------------
    # Context Relevancy Metric
    # --------------------------------------------------------------------------
    
    def compute_context_relevancy(
        self,
        question: str,
        contexts: List[str],
    ) -> float:
        """
        Compute context relevancy score.
        
        Measures how relevant the retrieved contexts are to the question.
        
        Process:
        1. For each context chunk, rate its relevancy to the question
        2. Context Relevancy = average of all chunk relevancy scores
        
        Args:
            question: The user question.
            contexts: List of retrieved context strings.
            
        Returns:
            float: Context relevancy score (0-1).
        """
        if not contexts:
            return 0.0
        
        relevancy_scores = []
        
        for context in contexts:
            prompt = CONTEXT_RELEVANCY_PROMPT.format(
                question=question,
                context=context,
            )
            
            response = self._call_llm(prompt, max_tokens=10)
            
            try:
                score = float(response.strip())
                score = max(0.0, min(1.0, score))
            except ValueError:
                score = 0.5  # Default if parsing fails
            
            relevancy_scores.append(score)
        
        return sum(relevancy_scores) / len(relevancy_scores)
    
    # --------------------------------------------------------------------------
    # Context Precision Metric
    # --------------------------------------------------------------------------
    
    def compute_context_precision(
        self,
        question: str,
        contexts: List[str],
    ) -> float:
        """
        Compute context precision score.
        
        Measures if relevant contexts appear earlier in the ranking.
        Uses Average Precision calculation.
        
        Process:
        1. For each context, determine if it's relevant (binary)
        2. Calculate precision at each relevant position
        3. Context Precision = average of precisions at relevant positions
        
        Args:
            question: The user question.
            contexts: List of retrieved context strings (ranked).
            
        Returns:
            float: Context precision score (0-1).
        """
        if not contexts:
            return 0.0
        
        # Determine relevancy for each context
        relevancy = []
        for context in contexts:
            prompt = CONTEXT_PRECISION_PROMPT.format(
                question=question,
                context=context,
            )
            
            verdict = self._call_llm(prompt, max_tokens=10)
            is_relevant = verdict.lower().strip() in ["yes", "true", "1"]
            relevancy.append(is_relevant)
        
        # Calculate Average Precision
        if not any(relevancy):
            return 0.0
        
        precisions = []
        relevant_count = 0
        
        for i, is_rel in enumerate(relevancy):
            if is_rel:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                precisions.append(precision_at_i)
        
        return sum(precisions) / len(precisions) if precisions else 0.0
    
    # --------------------------------------------------------------------------
    # Context Recall Metric
    # --------------------------------------------------------------------------
    
    def compute_context_recall(
        self,
        contexts: List[str],
        ground_truth: str,
    ) -> float:
        """
        Compute context recall score.
        
        Measures what fraction of the ground truth is captured by the contexts.
        
        Process:
        1. Extract statements from ground truth
        2. Check if each statement is supported by any context
        3. Context Recall = supported_statements / total_statements
        
        Args:
            contexts: List of retrieved context strings.
            ground_truth: The ground truth answer.
            
        Returns:
            float: Context recall score (0-1).
        """
        if not contexts or not ground_truth:
            return 0.0
        
        combined_contexts = "\n\n---\n\n".join(contexts)
        
        prompt = CONTEXT_RECALL_PROMPT.format(
            ground_truth=ground_truth,
            contexts=combined_contexts,
        )
        
        response = self._call_llm(prompt, max_tokens=500)
        result = self._extract_json(response)
        
        if not result or not isinstance(result, dict):
            return 0.0
        
        supported = result.get("supported", [])
        if not supported:
            return 0.0
        
        return sum(1 for s in supported if s) / len(supported)
    
    # --------------------------------------------------------------------------
    # Main Evaluation Method
    # --------------------------------------------------------------------------
    
    def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
        metrics: Optional[List[str]] = None,
    ) -> RagasMetrics:
        """
        Compute RAGAS metrics for a single sample.
        
        Args:
            question: The user question.
            answer: The generated answer.
            contexts: List of retrieved context strings.
            ground_truth: Optional ground truth answer (for recall).
            metrics: List of metrics to compute. If None, computes all available.
                     Options: "faithfulness", "answer_relevancy", "context_relevancy",
                              "context_precision", "context_recall"
        
        Returns:
            RagasMetrics: Computed metrics.
        """
        available_metrics = {
            "faithfulness": lambda: self.compute_faithfulness(question, answer, contexts),
            "answer_relevancy": lambda: self.compute_answer_relevancy(question, answer),
            "context_relevancy": lambda: self.compute_context_relevancy(question, contexts),
            "context_precision": lambda: self.compute_context_precision(question, contexts),
        }
        
        if ground_truth:
            available_metrics["context_recall"] = lambda: self.compute_context_recall(
                contexts, ground_truth
            )
        
        # Select metrics to compute
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
                print(f"Warning: Failed to compute {metric_name}: {e}")
        
        return results
    
    def evaluate_batch(
        self,
        samples: List[RagasInput],
        metrics: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate multiple samples and compute aggregate metrics.
        
        Args:
            samples: List of RagasInput samples.
            metrics: List of metrics to compute.
            verbose: Print progress.
            
        Returns:
            dict: Aggregate metrics and per-sample results.
        """
        all_results = []
        
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
        
        # Compute averages
        aggregates = {}
        if all_results:
            all_keys = set()
            for r in all_results:
                all_keys.update(r.keys())
            
            for key in all_keys:
                values = [r[key] for r in all_results if key in r and r[key] is not None]
                if values:
                    aggregates[f"avg_{key}"] = sum(values) / len(values)
        
        return {
            "aggregate_metrics": aggregates,
            "per_sample_metrics": all_results,
            "num_samples": len(samples),
        }


# ==============================================================================
# Convenience Functions
# ==============================================================================

def evaluate_ragas(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None,
    **kwargs,
) -> Dict[str, float]:
    """
    Convenience function to evaluate a single RAG response.
    
    Args:
        question: The user question.
        answer: The generated answer.
        contexts: List of retrieved context strings.
        ground_truth: Optional ground truth answer.
        **kwargs: Additional arguments for RagasEvaluator.
        
    Returns:
        dict: RAGAS metrics.
    """
    evaluator = RagasEvaluator(**kwargs)
    result = evaluator.evaluate(question, answer, contexts, ground_truth)
    return result.to_dict()


def create_ragas_dataset(
    eval_data: List[Dict[str, Any]],
) -> List[RagasInput]:
    """
    Convert evaluation data to RAGAS input format.
    
    Args:
        eval_data: List of dicts with question, answer, contexts, etc.
        
    Returns:
        list: List of RagasInput objects.
    """
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


# ==============================================================================
# Main Demo
# ==============================================================================

if __name__ == "__main__":
    # Demo usage
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
    else:
        print("Set GOOGLE_API_KEY environment variable for RAGAS evaluation")
