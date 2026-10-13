"""
Answer evaluation for RAG pipeline.

Evaluates faithfulness and relevancy of generated answers.

Usage:
    from evaluation.answer_eval import AnswerEvaluator
    
    evaluator = AnswerEvaluator()
    scores = evaluator.evaluate(question, answer, context)
"""

import os
from typing import List, Dict, Any, Optional

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False


# Evaluation prompts
FAITHFULNESS_PROMPT = """Evaluate if the answer is faithful to the given context.
The answer should only contain information that can be directly inferred from the context.

Context:
{context}

Question: {question}

Answer: {answer}

Score the faithfulness from 0 to 1:
- 1.0: Completely faithful, all claims are supported by context
- 0.5: Partially faithful, some claims not in context
- 0.0: Not faithful, contains hallucinations or contradictions

Respond with just the score (a number between 0 and 1)."""

RELEVANCY_PROMPT = """Evaluate if the answer is relevant to the question.

Question: {question}

Answer: {answer}

Score the relevancy from 0 to 1:
- 1.0: Directly answers the question
- 0.5: Partially relevant
- 0.0: Not relevant or off-topic

Respond with just the score (a number between 0 and 1)."""


class AnswerEvaluator:
    """Evaluator for answer quality metrics."""
    
    def __init__(
        self,
        model_name: str = "gpt-4",
        api_key: Optional[str] = None,
    ):
        """
        Initialize the evaluator.
        
        Args:
            model_name: LLM model for evaluation.
            api_key: API key (defaults to OPENAI_API_KEY env var).
        """
        self.model_name = model_name
        self.client = None
        
        if HAS_OPENAI:
            api_key = api_key or os.getenv("OPENAI_API_KEY")
            if api_key:
                self.client = OpenAI(api_key=api_key)
    
    def _call_llm(self, prompt: str) -> float:
        """Call LLM and parse score."""
        if not self.client:
            return 0.5  # Default score if no LLM available
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        
        try:
            score = float(response.choices[0].message.content.strip())
            return max(0.0, min(1.0, score))
        except (ValueError, AttributeError):
            return 0.5
    
    def evaluate_faithfulness(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> float:
        """
        Evaluate if the answer is faithful to the context.
        
        Args:
            question: User question.
            answer: Generated answer.
            context: Retrieved context.
            
        Returns:
            float: Faithfulness score (0-1).
        """
        prompt = FAITHFULNESS_PROMPT.format(
            context=context,
            question=question,
            answer=answer,
        )
        return self._call_llm(prompt)
    
    def evaluate_relevancy(
        self,
        question: str,
        answer: str,
    ) -> float:
        """
        Evaluate if the answer is relevant to the question.
        
        Args:
            question: User question.
            answer: Generated answer.
            
        Returns:
            float: Relevancy score (0-1).
        """
        prompt = RELEVANCY_PROMPT.format(
            question=question,
            answer=answer,
        )
        return self._call_llm(prompt)
    
    def evaluate(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> Dict[str, float]:
        """
        Evaluate answer quality.
        
        Args:
            question: User question.
            answer: Generated answer.
            context: Retrieved context.
            
        Returns:
            dict: Evaluation scores.
        """
        return {
            "faithfulness": self.evaluate_faithfulness(question, answer, context),
            "relevancy": self.evaluate_relevancy(question, answer),
        }
    
    def evaluate_batch(
        self,
        samples: List[Dict[str, str]],
    ) -> Dict[str, float]:
        """
        Evaluate multiple samples and compute averages.
        
        Args:
            samples: List of dicts with question, answer, context.
            
        Returns:
            dict: Average scores.
        """
        all_scores = []
        
        for sample in samples:
            scores = self.evaluate(
                sample["question"],
                sample["answer"],
                sample["context"],
            )
            all_scores.append(scores)
        
        # Compute averages
        avg_scores = {}
        if all_scores:
            for key in all_scores[0].keys():
                avg_scores[f"avg_{key}"] = sum(s[key] for s in all_scores) / len(all_scores)
        
        avg_scores["num_samples"] = len(samples)
        return avg_scores


def evaluate_answer(
    question: str,
    answer: str,
    context: str,
    **kwargs,
) -> Dict[str, float]:
    """
    Convenience function to evaluate an answer.
    
    Args:
        question: User question.
        answer: Generated answer.
        context: Retrieved context.
        **kwargs: Additional arguments for AnswerEvaluator.
        
    Returns:
        dict: Evaluation scores.
    """
    evaluator = AnswerEvaluator(**kwargs)
    return evaluator.evaluate(question, answer, context)


if __name__ == "__main__":
    # Demo usage
    sample = {
        "question": "What is the maintenance schedule?",
        "answer": "The maintenance schedule requires checking oil levels every 1000 hours.",
        "context": "The maintenance schedule requires checking oil levels every 1000 hours. All filters should be replaced during regular maintenance intervals.",
    }
    
    evaluator = AnswerEvaluator()
    
    if evaluator.client:
        scores = evaluator.evaluate(**sample)
        print("Answer Evaluation:")
        print(f"  Faithfulness: {scores['faithfulness']:.2f}")
        print(f"  Relevancy: {scores['relevancy']:.2f}")
    else:
        print("Set OPENAI_API_KEY for LLM-based evaluation")
        print("Using default scores: 0.5")
