"""
Vector retriever for RAG pipeline.

Runs vector similarity search against ChromaDB.

Usage:
    from src.retrieval.retriever import Retriever
    
    retriever = Retriever(vector_store_dir="data/vectordb")
    results = retriever.search("What is the maintenance schedule?", top_k=5)
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VectorStore:
    """ChromaDB-based vector store for document chunks."""
    
    def __init__(
        self,
        persist_dir: str | Path,
        collection_name: str = "documents",
    ):
        """
        Initialize the vector store.
        
        Args:
            persist_dir: Directory to persist the database.
            collection_name: Name of the collection.
        """
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )
        
        self.collection_name = collection_name
        self.collection = None
    
    def get_or_create_collection(
        self,
        embedding_dimension: Optional[int] = None,
    ):
        """Get or create the collection."""
        metadata = None
        if embedding_dimension:
            metadata = {"embedding_dimension": embedding_dimension}
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata=metadata,
        )
        return self.collection
    
    def delete_collection(self):
        """Delete the collection if it exists."""
        try:
            self.client.delete_collection(self.collection_name)
            self.collection = None
            print(f"Deleted collection: {self.collection_name}")
        except (ValueError, chromadb.errors.NotFoundError):
            print(f"Collection {self.collection_name} does not exist")
    
    def index_embeddings(
        self,
        embeddings_path: Path,
        batch_size: int = 100,
    ) -> int:
        """
        Index embeddings from a JSON file.
        
        Args:
            embeddings_path: Path to embeddings JSON file.
            batch_size: Batch size for indexing.
            
        Returns:
            int: Number of chunks indexed.
        """
        data = json.loads(embeddings_path.read_text(encoding="utf-8"))
        
        document_id = data["document_id"]
        embedding_dimension = data.get("embedding_dimension")
        chunks = data["chunks"]
        
        if self.collection is None:
            self.get_or_create_collection(embedding_dimension)
        
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            chunk_id = chunk["chunk_id"]
            
            # Skip if already indexed
            existing = self.collection.get(ids=[chunk_id])
            if existing["ids"]:
                continue
            
            ids.append(chunk_id)
            embeddings.append(chunk["embedding"])
            documents.append(chunk.get("contextualized_text", chunk["text"]))
            
            metadata = {
                "document_id": document_id,
                "text": chunk["text"][:1000],
                "content_type": chunk.get("content_type", "text"),
            }
            
            if chunk.get("page_start") is not None:
                metadata["page_start"] = chunk["page_start"]
            if chunk.get("page_end") is not None:
                metadata["page_end"] = chunk["page_end"]
            if chunk.get("headings"):
                metadata["headings"] = " > ".join(chunk["headings"])
            
            metadatas.append(metadata)
        
        # Index in batches
        indexed = 0
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_documents = documents[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            
            self.collection.add(
                ids=batch_ids,
                embeddings=batch_embeddings,
                documents=batch_documents,
                metadatas=batch_metadatas,
            )
            indexed += len(batch_ids)
        
        return indexed
    
    def index_all(
        self,
        embeddings_dir: Path,
        reset: bool = False,
    ) -> int:
        """Index all embeddings from a directory."""
        embeddings_dir = Path(embeddings_dir)
        
        if reset:
            self.delete_collection()
        
        embedding_files = sorted(embeddings_dir.glob("*.json"))
        
        if not embedding_files:
            print(f"No embedding files found in: {embeddings_dir}")
            return 0
        
        print(f"Found {len(embedding_files)} files to index.")
        
        total = 0
        for index, emb_file in enumerate(embedding_files, start=1):
            print(f"[{index}/{len(embedding_files)}] Indexing: {emb_file.name}")
            num = self.index_embeddings(emb_file)
            print(f"  -> {num} chunks indexed")
            total += num
        
        print("=" * 60)
        print(f"Indexing complete. Total: {total}")
        print("=" * 60)
        
        return total
    
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.
        
        Args:
            query_embedding: Query embedding vector.
            top_k: Number of results.
            where: Filter conditions.
            
        Returns:
            list: Search results.
        """
        if self.collection is None:
            self.get_or_create_collection()
        
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        
        formatted = []
        for i in range(len(results["ids"][0])):
            formatted.append({
                "chunk_id": results["ids"][0][i],
                "document": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "score": 1 - results["distances"][0][i],
            })
        
        return formatted
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        if self.collection is None:
            self.get_or_create_collection()
        
        all_data = self.collection.get(include=["metadatas"])
        doc_ids = set()
        content_types = {}
        
        for metadata in all_data["metadatas"]:
            doc_ids.add(metadata.get("document_id", "unknown"))
            ct = metadata.get("content_type", "text")
            content_types[ct] = content_types.get(ct, 0) + 1
        
        return {
            "collection_name": self.collection_name,
            "total_chunks": self.collection.count(),
            "total_documents": len(doc_ids),
            "document_ids": sorted(doc_ids),
            "content_types": content_types,
        }


class Retriever:
    """Retriever for searching document chunks."""
    
    def __init__(
        self,
        vector_store_dir: str | Path,
        model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "documents",
    ):
        """
        Initialize the retriever.
        
        Args:
            vector_store_dir: Path to vector store directory.
            model_name: Embedding model name.
            collection_name: Collection name.
        """
        self.vector_store = VectorStore(
            persist_dir=vector_store_dir,
            collection_name=collection_name,
        )
        self.vector_store.get_or_create_collection()
        
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"Model loaded. Dimension: {self.model.get_embedding_dimension()}")
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        document_id: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query.
            top_k: Number of results.
            document_id: Filter by document.
            content_type: Filter by content type.
            
        Returns:
            list: Search results.
        """
        # Build filter
        where = None
        if document_id or content_type:
            where = {}
            if document_id:
                where["document_id"] = document_id
            if content_type:
                where["content_type"] = content_type
        
        # Encode query
        query_embedding = self.model.encode(query).tolist()
        
        return self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
            where=where,
        )
    
    def format_results(
        self,
        results: List[Dict[str, Any]],
        show_score: bool = True,
    ) -> str:
        """Format search results for display."""
        if not results:
            return "No results found."
        
        lines = []
        for i, result in enumerate(results, start=1):
            lines.append(f"\n{'='*60}")
            lines.append(f"Result {i}: {result['chunk_id']}")
            
            if show_score:
                lines.append(f"Score: {result['score']:.4f}")
            
            meta = result["metadata"]
            lines.append(f"Document: {meta.get('document_id', 'N/A')}")
            lines.append(f"Pages: {meta.get('page_start', '?')}-{meta.get('page_end', '?')}")
            
            lines.append("-" * 60)
            lines.append(result["document"][:500])
            if len(result["document"]) > 500:
                lines.append("...")
        
        return "\n".join(lines)


if __name__ == "__main__":
    VECTOR_STORE_DIR = Path(__file__).parent.parent.parent / "data" / "vectordb"
    
    if VECTOR_STORE_DIR.exists():
        retriever = Retriever(VECTOR_STORE_DIR)
        
        queries = [
            "What is the maintenance schedule?",
            "What are the pressure specifications?",
        ]
        
        for query in queries:
            print(f"\n{'='*70}")
            print(f"QUERY: {query}")
            print("=" * 70)
            
            results = retriever.search(query, top_k=3)
            print(retriever.format_results(results))
    else:
        print(f"Directory not found: {VECTOR_STORE_DIR}")
