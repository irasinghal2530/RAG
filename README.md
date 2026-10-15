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
│   │   ├── chunker.py             # Main chunking logic
│   │   ├── overlap_manager.py     # Handles chunk overlap
│   │   └── metadata_tagger.py     # Tags chunks with context
│   │
│   ├── embedding/                  # Embedding generation
│   │   └── base_embedder.py       # Local sentence-transformers embeddings
│   │
│   ├── retrieval/                  # Vector search
│   │   ├── retriever.py           # Vector similarity search
│   │   └── hybrid_retriever.py    # BM25 + dense fusion
│   │
│   ├── reranking/                  # Result reranking
│   │   ├── base_reranker.py       # Cross-encoder reranker
│   │   └── cohere_reranker.py     # Cohere rerank API client
│   │
│   └── generation/                 # LLM response generation
│       ├── prompt_builder.py      # Builds RAG prompts
│       └── llm_client.py          # Google Gemini API client
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

# Edit .env with your Gemini key
# GOOGLE_API_KEY=...  (https://aistudio.google.com/apikey)
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
builder = PromptBuilder()
client = LLMClient()

messages = builder.build_messages("What is the maintenance schedule?", results)
response = client.generate(messages)

print(response["content"])
```

### Run Evaluation

```bash
python scripts/run_eval.py --eval-file data/eval_set.json
```

## 🔧 Configuration

### LLM Config (`config/llm_config.yaml`)

```yaml
llm:
  provider: "gemini"
  model_name: "gemini-2.0-flash"
  temperature: 0.0
  max_tokens: 1024
```

### Retriever Config (`config/retriever_config.yaml`)

```yaml
embedding:
  model_name: "all-MiniLM-L6-v2"
  
retrieval:
  top_k: 5
  
reranking:
  enabled: false
  model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
```

## 🐳 Docker

```bash
# Build image
docker build -t rag-system .

# Run with mounted data
docker run -v $(pwd)/data:/app/data rag-system
```

## 📊 Evaluation Metrics

- **Hit Rate@K**: Proportion of relevant docs in top K
- **MRR**: Mean Reciprocal Rank
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **Faithfulness**: Is the answer grounded in context?
- **Relevancy**: Does the answer address the question?

## 📝 License

MIT License
