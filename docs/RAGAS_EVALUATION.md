# RAGAS Evaluation for RAG Pipeline

## Overview

RAGAS (Retrieval Augmented Generation Assessment) is a comprehensive framework for evaluating RAG pipelines. This implementation provides automated assessment of both retrieval and generation quality using LLM-as-a-judge methodology.

**Reference Paper:** [RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217)

---

## Table of Contents

1. [Metrics Overview](#metrics-overview)
2. [Architecture](#architecture)
3. [Folder Structure](#folder-structure)
4. [Integration with Existing Code](#integration-with-existing-code)
5. [Usage Guide](#usage-guide)
6. [Evaluation Dataset Format](#evaluation-dataset-format)
7. [Metrics Deep Dive](#metrics-deep-dive)
8. [Results Interpretation](#results-interpretation)
9. [Best Practices](#best-practices)

---

## Metrics Overview

| Metric | Category | Requires Ground Truth | Description |
|--------|----------|----------------------|-------------|
| **Faithfulness** | Answer Quality | No | Is the answer grounded in the retrieved context? |
| **Answer Relevancy** | Answer Quality | No | Does the answer address the question? |
| **Context Relevancy** | Retrieval Quality | No | Are retrieved contexts relevant to the question? |
| **Context Precision** | Retrieval Quality | No | Are relevant contexts ranked higher? |
| **Context Recall** | Retrieval Quality | Yes | Does context cover the ground truth answer? |

### Metric Scores

All metrics return scores between **0** and **1**:
- **1.0** = Perfect score
- **0.5** = Moderate performance
- **0.0** = Poor performance

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RAGAS Evaluation Flow                              │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │  Evaluation Set │
                              │    (JSON)       │
                              └────────┬────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │         RAG Pipeline                  │
                    │  ┌───────────┐    ┌───────────────┐  │
                    │  │ Retriever │───▶│ LLM Generator │  │
                    │  └───────────┘    └───────────────┘  │
                    └──────────────────────────────────────┘
                                       │
                           ┌───────────┴───────────┐
                           │                       │
                           ▼                       ▼
                    ┌─────────────┐         ┌─────────────┐
                    │  Contexts   │         │   Answer    │
                    │ (Retrieved) │         │ (Generated) │
                    └──────┬──────┘         └──────┬──────┘
                           │                       │
                           └───────────┬───────────┘
                                       │
                                       ▼
                    ┌──────────────────────────────────────┐
                    │         RAGAS Evaluator              │
                    │                                      │
                    │  ┌────────────────┐ ┌─────────────┐  │
                    │  │  Faithfulness  │ │  Answer     │  │
                    │  │                │ │  Relevancy  │  │
                    │  └────────────────┘ └─────────────┘  │
                    │                                      │
                    │  ┌────────────────┐ ┌─────────────┐  │
                    │  │    Context     │ │  Context    │  │
                    │  │   Relevancy    │ │  Precision  │  │
                    │  └────────────────┘ └─────────────┘  │
                    │                                      │
                    │  ┌────────────────┐                  │
                    │  │    Context     │                  │
                    │  │    Recall      │                  │
                    │  └────────────────┘                  │
                    └──────────────────────────────────────┘
                                       │
                                       ▼
                              ┌─────────────────┐
                              │  RAGAS Metrics  │
                              │    (Output)     │
                              └─────────────────┘
```

---

## Folder Structure

```
study_rag/
├── main.py                          # Entry point with 'ragas' command
├── requirements.txt                 # Dependencies
│
├── evaluation/
│   ├── __init__.py                  # Module exports
│   ├── ragas_eval.py               # ★ RAGAS implementation
│   ├── retrieval_eval.py           # Retrieval metrics (MRR, NDCG, etc.)
│   └── answer_eval.py              # Basic answer evaluation
│
├── scripts/
│   ├── run_ragas_eval.py           # ★ RAGAS evaluation script
│   ├── run_eval.py                 # Basic evaluation script
│   └── build_index.py              # Index builder
│
├── src/
│   ├── retrieval/
│   │   ├── retriever.py            # Vector store retrieval
│   │   └── hybrid_retriever.py     # Hybrid search
│   │
│   ├── generation/
│   │   ├── llm_client.py           # LLM API client
│   │   └── prompt_builder.py       # Prompt construction
│   │
│   ├── embedding/                  # Embedding generation
│   ├── chunking/                   # Document chunking
│   └── ingestion/                  # Document loading
│
├── data/
│   ├── eval_set_example.json       # ★ Example evaluation dataset
│   ├── raw/                        # Raw documents
│   ├── processed/                  # Processed chunks
│   └── vectordb/                   # Vector database
│
└── docs/
    └── RAGAS_EVALUATION.md         # ★ This documentation
```

**Key files for RAGAS:**
- `evaluation/ragas_eval.py` - Core RAGAS metrics implementation
- `scripts/run_ragas_eval.py` - Evaluation runner script
- `data/eval_set_example.json` - Example evaluation dataset format

---

## Integration with Existing Code

### How RAGAS Connects to Other Modules

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        Integration Diagram                                │
└──────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌──────────────────┐     ┌─────────────────────────┐
│   main.py       │────▶│ run_ragas_eval.py│────▶│   ragas_eval.py         │
│   (CLI entry)   │     │ (orchestration)  │     │   (metrics logic)       │
└─────────────────┘     └────────┬─────────┘     └─────────────────────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼            ▼
            ┌───────────┐ ┌───────────┐ ┌──────────────┐
            │ Retriever │ │ LLMClient │ │PromptBuilder │
            │           │ │           │ │              │
            │(retrieval/│ │(generation│ │(generation/  │
            │ retriever │ │/ llm_     │ │ prompt_      │
            │ .py)      │ │ client.py)│ │ builder.py)  │
            └───────────┘ └───────────┘ └──────────────┘
```

### Code Dependencies

**ragas_eval.py imports:**
```python
# External
from google import genai                     # Gemini LLM for evaluation
from sentence_transformers import SentenceTransformer  # Embeddings

# No internal imports - standalone evaluation module
```

**run_ragas_eval.py imports:**
```python
# Internal
from src.retrieval import Retriever          # Get relevant chunks
from src.generation import PromptBuilder, LLMClient  # Generate answers
from evaluation.ragas_eval import RagasEvaluator, RagasInput  # RAGAS metrics
```

### Reused Components

| Component | Used By RAGAS | Purpose |
|-----------|---------------|---------|
| `Retriever` | Yes | Retrieve relevant document chunks |
| `LLMClient` | Yes | Generate answers for evaluation |
| `PromptBuilder` | Yes | Build prompts for answer generation |
| `VectorStore` | Indirect | Stores embeddings (used by Retriever) |
| `SentenceTransformer` | Yes | Compute embeddings for answer relevancy |

---

## Usage Guide

### Command Line Usage

#### Run Full RAGAS Evaluation
```bash
python main.py ragas --eval-file data/eval_set_example.json
```

#### Select Specific Metrics
```bash
# Only compute faithfulness and context precision
python main.py ragas --eval-file data/eval_set.json \
    --metrics faithfulness context_precision
```

#### Save Results to File
```bash
python main.py ragas --eval-file data/eval_set.json \
    --output results/ragas_results.json
```

#### Skip Answer Generation (Retrieval Metrics Only)
```bash
python main.py ragas --eval-file data/eval_set.json \
    --skip-generation
```

#### Configure Retrieval
```bash
python main.py ragas --eval-file data/eval_set.json \
    --vectordb-dir data/custom_vectordb \
    --top-k 10
```

### Programmatic Usage

```python
from evaluation.ragas_eval import RagasEvaluator, RagasInput

# Initialize evaluator
evaluator = RagasEvaluator(model_name="gemini-2.0-flash")

# Prepare input
sample = RagasInput(
    question="What is the maintenance schedule?",
    answer="Oil should be checked every 1000 hours.",
    contexts=[
        "The maintenance schedule requires checking oil levels every 1000 hours.",
        "All filters should be replaced during regular intervals.",
    ],
    ground_truth="Oil checks every 1000 hours, filters replaced regularly."
)

# Compute all metrics
metrics = evaluator.evaluate(
    question=sample.question,
    answer=sample.answer,
    contexts=sample.contexts,
    ground_truth=sample.ground_truth,
)

print(metrics)
# RagasMetrics(faithfulness=0.95, answer_relevancy=0.87, ...)
```

### Batch Evaluation

```python
from evaluation.ragas_eval import RagasEvaluator, create_ragas_dataset
import json

# Load evaluation data
with open("data/eval_set.json") as f:
    eval_data = json.load(f)

# Convert to RAGAS format
samples = create_ragas_dataset(eval_data)

# Evaluate batch
evaluator = RagasEvaluator()
results = evaluator.evaluate_batch(samples)

print(f"Average Faithfulness: {results['aggregate_metrics']['avg_faithfulness']:.4f}")
```

---

## Evaluation Dataset Format

### Required JSON Structure

```json
[
  {
    "question": "What is the maintenance schedule?",
    "ground_truth": "Oil must be checked every 1000 hours. Filters need replacement.",
    "expected_answer": "Oil checks every 1000 hours.",
    "relevant_chunk_ids": ["MNT-001_chunk_001", "MNT-001_chunk_002"]
  }
]
```

### Field Descriptions

| Field | Required | Description |
|-------|----------|-------------|
| `question` | Yes | The user's question |
| `ground_truth` | No* | Reference answer for recall metrics |
| `expected_answer` | No | Alternative field for ground truth |
| `relevant_chunk_ids` | No | For retrieval evaluation metrics |

*Ground truth is required for `context_recall` metric.

### Example Dataset

See `data/eval_set_example.json` for a complete example.

---

## Metrics Deep Dive

### 1. Faithfulness

**Question:** Is the generated answer grounded in the retrieved context?

**Process:**
1. Extract atomic factual statements from the answer
2. For each statement, verify if it can be inferred from context
3. Compute: `supported_statements / total_statements`

**LLM Prompts Used:**
- Statement extraction prompt
- Statement verification prompt

**Example:**
```
Context: "Oil must be checked every 1000 hours."
Answer: "Check oil every 1000 hours and replace filters monthly."

Statements:
  1. "Check oil every 1000 hours" → Supported ✓
  2. "Replace filters monthly" → NOT supported ✗

Faithfulness = 1/2 = 0.5
```

**When to use:** Always - hallucination detection is critical.

---

### 2. Answer Relevancy

**Question:** Does the answer actually address the question?

**Process:**
1. Generate N reverse questions that the answer might be answering
2. Compute embedding similarity between original and generated questions
3. Average the similarities

**Uses:** Sentence Transformers for embedding similarity

**Example:**
```
Question: "What is the maintenance schedule?"
Answer: "Oil should be checked every 1000 hours."

Generated reverse questions:
  1. "How often should oil be checked?" (similarity: 0.85)
  2. "What is the oil maintenance interval?" (similarity: 0.82)
  3. "When should I check the oil?" (similarity: 0.78)

Answer Relevancy = (0.85 + 0.82 + 0.78) / 3 = 0.817
```

**When to use:** When concerned about off-topic or tangential answers.

---

### 3. Context Relevancy

**Question:** Are the retrieved contexts relevant to the question?

**Process:**
1. For each context chunk, rate its relevancy (0-1)
2. Average all chunk relevancy scores

**Example:**
```
Question: "What is the maintenance schedule?"

Context 1: "Oil must be checked every 1000 hours." → 1.0
Context 2: "Company was founded in 1985." → 0.0
Context 3: "Maintenance intervals are important." → 0.5

Context Relevancy = (1.0 + 0.0 + 0.5) / 3 = 0.5
```

**When to use:** To evaluate retriever quality.

---

### 4. Context Precision

**Question:** Are relevant contexts ranked higher in the results?

**Process:**
1. For each context, determine if it's relevant (binary)
2. Calculate Average Precision based on rank positions

**Example:**
```
Contexts (ranked): [relevant, irrelevant, relevant, irrelevant]

Precision at position 1: 1/1 = 1.0 (hit)
Precision at position 3: 2/3 = 0.67 (hit)

Context Precision = (1.0 + 0.67) / 2 = 0.835
```

**When to use:** To optimize retriever ranking.

---

### 5. Context Recall

**Question:** Does the context cover all information in the ground truth?

**Requires:** Ground truth answer

**Process:**
1. Extract statements from ground truth
2. Check if each statement is supported by any context
3. Compute: `supported_statements / total_statements`

**Example:**
```
Ground Truth: "Oil checks every 1000 hours. Filters need replacement."

Statements:
  1. "Oil checks every 1000 hours" → Found in context ✓
  2. "Filters need replacement" → NOT found ✗

Context Recall = 1/2 = 0.5
```

**When to use:** When you have reference answers available.

---

## Results Interpretation

### Sample Output

```
============================================================
RAGAS Evaluation Results
============================================================

Aggregate Metrics:
----------------------------------------
  Faithfulness (answer grounded in context)
    ████████████████░░░░ 0.8234

  Answer Relevancy (answer addresses question)
    ███████████████░░░░░ 0.7650

  Context Relevancy (contexts are relevant)
    ██████████████░░░░░░ 0.7100

  Context Precision (relevant contexts ranked higher)
    █████████████████░░░ 0.8500

  Context Recall (context covers ground truth)
    ██████████████████░░ 0.9200

Total samples evaluated: 25
```

### Score Benchmarks

| Score Range | Quality Level | Action |
|-------------|---------------|--------|
| 0.90 - 1.00 | Excellent | Maintain current approach |
| 0.75 - 0.89 | Good | Minor optimizations may help |
| 0.50 - 0.74 | Moderate | Consider improvements |
| 0.25 - 0.49 | Poor | Significant changes needed |
| 0.00 - 0.24 | Critical | Major overhaul required |

### Metric-Specific Guidance

| Low Metric | Possible Causes | Fixes |
|------------|-----------------|-------|
| **Faithfulness < 0.7** | LLM hallucinations | Stricter prompts, better context grounding |
| **Answer Relevancy < 0.7** | Off-topic answers | Improve prompt engineering |
| **Context Relevancy < 0.7** | Poor retrieval | Better embeddings, reranking |
| **Context Precision < 0.7** | Bad ranking | Add reranker, tune retriever |
| **Context Recall < 0.7** | Missing information | Increase top_k, improve chunking |

---

## Best Practices

### 1. Evaluation Dataset

- **Size:** Minimum 25-50 samples for reliable metrics
- **Diversity:** Cover different question types and topics
- **Ground truth:** Include when possible for recall metrics
- **Regular updates:** Refresh as your document corpus changes

### 2. Model Selection

- Use **gemini-2.0-flash** (free tier, default)
- Use **gemini-2.5-pro** if you need stronger judging accuracy

### 3. Running Evaluations

```bash
# Development: Quick check with fewer samples
python main.py ragas --eval-file data/eval_dev.json --metrics faithfulness

# Production: Full evaluation with all metrics
python main.py ragas --eval-file data/eval_full.json --output results/eval_$(date +%Y%m%d).json
```

### 4. CI/CD Integration

```yaml
# Example GitHub Actions workflow
name: RAG Evaluation
on: [push]
jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run RAGAS Evaluation
        run: |
          python main.py ragas --eval-file data/eval_set.json \
            --output results/ragas_results.json
      - name: Check thresholds
        run: python scripts/check_metrics.py results/ragas_results.json
```

### 5. Metric Thresholds

Set minimum acceptable scores:
```python
THRESHOLDS = {
    "avg_faithfulness": 0.80,
    "avg_answer_relevancy": 0.75,
    "avg_context_relevancy": 0.70,
    "avg_context_precision": 0.75,
    "avg_context_recall": 0.70,
}
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `Google API key not found` | Set `GOOGLE_API_KEY` in `.env` |
| `No embedding model` | Install `sentence-transformers` |
| `Empty contexts` | Check retriever configuration |
| `Low scores across all metrics` | Verify eval dataset quality |

### Debug Mode

```python
# Enable verbose output
evaluator = RagasEvaluator(model_name="gemini-2.0-flash")

# Test single sample
result = evaluator.evaluate(
    question="Test question",
    answer="Test answer",
    contexts=["Test context"],
    ground_truth="Test ground truth",
)

print(f"Faithfulness: {result.faithfulness}")
print(f"Answer Relevancy: {result.answer_relevancy}")
```

---

## References

1. RAGAS Paper: [https://arxiv.org/abs/2309.15217](https://arxiv.org/abs/2309.15217)
2. Official RAGAS Library: [https://github.com/explodinggradients/ragas](https://github.com/explodinggradients/ragas)
3. LLM-as-a-Judge: [https://arxiv.org/abs/2306.05685](https://arxiv.org/abs/2306.05685)
