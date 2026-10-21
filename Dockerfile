# RAG Production System Dockerfile
# Build: docker build -t rag-system .
# Run:   docker run -v $(pwd)/data:/app/data rag-system

FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY config/ config/
COPY src/ src/
COPY evaluation/ evaluation/
COPY scripts/ scripts/
COPY main.py .

# Create data directories
RUN mkdir -p data/raw data/processed data/vectordb

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "-c", "print('RAG System Ready. Use scripts/ingest_documents.sh to start.')"]
