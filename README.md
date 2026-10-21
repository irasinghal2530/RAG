# RAG Production System

A production-ready Retrieval-Augmented Generation (RAG) pipeline for document processing, embedding, retrieval, and generation.

## 📁 Project Structure

```
RAG_PRODUCTION_SYSTEM/
├── config/                          # Configuration files
│   ├── llm_config.yaml             # LLM settings
│   └── retriever_config.yaml       # Retrieval parameters
│
├── data/                           # Data directories
│   ├── raw/                        # Original source documents (PDFs)
│   ├── processed/                  # Cleaned & chunked docs
│   │   ├── docling/               # Parsed document JSONs & Markdown
│   │   ├── chunks/                # Document chunks
│   │   └── embeddings/            # Chunk embeddings
│   └── vectordb/                   # Vector DB indexes (ChromaDB)
│
├── src/                            # Source code
│   ├── ingestion/                  # Document loading & parsing
│   │   ├── loader.py              # Loads PDFs, URLs, docs
│   │   └── splitter.py            # Splits docs into sections
│   │
│   ├── chunking/                   # Chunking strategies
│   │   └── chunker.py             # Main chunking logic
│   │
│   ├── embedding/                  # Embedding generation
│   │   └── base_embedder.py       # Local sentence-transformers embeddings
│   │
│   ├── retrieval/                  # Vector search
│   │   ├── retriever.py           # Vector similarity search
│   │   └── hybrid_retriever.py    # BM25 + dense fusion
│   │
│   ├── reranking/                  # Result reranking
│   │   └── base_reranker.py       # Cross-encoder reranker
│   │
│   └── generation/                 # LLM response generation
│       ├── prompt_builder.py      # Builds RAG prompts
│       └── llm_client.py          # Groq API client (generator + judge)
│
├── evaluation/                     # Evaluation metrics
│   ├── retrieval_eval.py          # Hit rate, MRR, NDCG metrics
│   └── answer_eval.py             # Faithfulness & relevancy scores
│
├── scripts/                        # Utility scripts
│   ├── ingest_documents.sh        # Runs full ingestion pipeline
│   ├── build_index.py             # Builds vector DB index
│   └── run_eval.py                # Executes evaluation suite
│
├── .env.example                    # Environment variables template
├── Dockerfile                      # Container configuration
├── requirements.txt                # Dependencies
└── main.py                         # Main entry point
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone and enter directory
cd study_rag

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your Groq key (generator + RAGAS judge)
# GROQ_API_KEY=...  (https://console.groq.com/keys)
```

### 3. Add Documents

Place your PDF documents in `data/raw/`:

```bash
cp your_documents/*.pdf data/raw/
```

### 4. Run Ingestion Pipeline

```bash
# Full pipeline: Load → Parse → Chunk → Embed → Index
python main.py ingest

# Or with options
python main.py ingest --reset  # Reset vector store
```

### 5. Retrieve From Your Knowledge Base

```bash
python main.py retrieve "What is the maintenance schedule?"
```

### 6. Ask a Question (full RAG)

```bash
python main.py ask "What is the maintenance schedule?"
```

Retrieves the top chunks, sends them to the LLM, and prints the answer with numbered sources.

## 📚 Usage Examples

### Python API

```python
from src.retrieval import Retriever
from src.generation import LLMClient, PromptBuilder

# Initialize retriever
retriever = Retriever("data/vectordb")

# Search for relevant chunks
results = retriever.search("What is the maintenance schedule?", top_k=5)

# Generate response
client = LLMClient()
response = client.generate_with_context("What is the maintenance schedule?", results)

print(response["content"])
print(response["sources"])   # chunks that actually reached the LLM, in [Source n] order
```

### Run Evaluation

```bash
python main.py eval --eval-file data/eval_set_v2.json --output results/eval.json
```

## 🔧 Configuration

### LLM Config (`config/llm_config.yaml`)

```yaml
llm:
  provider: "groq"
  model_name: "openai/gpt-oss-20b"         # generator
  judge_model_name: "openai/gpt-oss-120b"  # RAGAS judge (LLM_JUDGE_MODEL in .env overrides)
  temperature: 0.0
  max_retries: 5                           # backoff on Groq rate limits
```

### Retriever Config (`config/retriever_config.yaml`)

These values are loaded by `src/config.py`; the Retriever and eval scripts use them.

```yaml
embedding:
  model_name: "all-MiniLM-L6-v2"
  normalize: true

vector_store:
  distance_metric: "cosine"   # re-ingest with --reset after changing

retrieval:
  top_k: 5          # chunks sent to the LLM
  candidate_k: 20   # first-stage candidates before reranking
  hybrid:
    enabled: true   # BM25 + dense, merged with Reciprocal Rank Fusion

reranking:
  enabled: true
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

Document titles shown to the retriever and the LLM live in `config/document_titles.yaml`.

### RAGAS Evaluation

```bash
python main.py ragas --eval-file data/eval_set_v2.json --output results/ragas.json
python main.py ragas --eval-file data/eval_set_v2.json --no-hybrid --no-rerank   # ablation
```

Metrics the judge could not parse are reported as failures and left out of the averages, not scored 0.

## 🐳 Docker

```bash
# Build image
docker build -t rag-system .

# Run with mounted data
docker run -v $(pwd)/data:/app/data rag-system
```

## 📊 Evaluation Metrics

- **Hit Rate@K**: Share of queries with at least one relevant chunk in the top K
- **Recall@K**: Share of each query's relevant chunks found in the top K
- **MRR**: Mean Reciprocal Rank of the first relevant chunk
- **NDCG@K**: Normalized Discounted Cumulative Gain (ideal = all relevant chunks ranked first)

Results produced before Oct 2026 labelled Recall@K as `hit_rate@K`.
- **Faithfulness**: Is the answer grounded in context?
- **Relevancy**: Does the answer address the question?

## 📝 License

MIT License
