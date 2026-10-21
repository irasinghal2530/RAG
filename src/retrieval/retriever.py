"""
Retriever for RAG pipeline.

Dense vector search against ChromaDB, optionally fused with BM25 keyword
search and reranked by a cross-encoder (both switched in
config/retriever_config.yaml).

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

from src.config import load_retriever_config
from src.retrieval.hybrid_retriever import BM25Index, reciprocal_rank_fusion


class VectorStore:
    """ChromaDB-based vector store for document chunks."""

    def __init__(
        self,
        persist_dir: str | Path,
        collection_name: Optional[str] = None,
        distance_metric: Optional[str] = None,
    ):
        """
        Initialize the vector store.

        Args:
            persist_dir: Directory to persist the database.
            collection_name: Name of the collection (default: retriever_config.yaml).
            distance_metric: hnsw space (default: retriever_config.yaml).
        """
        # Ingest and Retriever both build a VectorStore; reading the defaults
        # from config keeps them pointed at the same collection.
        store_cfg = load_retriever_config().get("vector_store", {})
        collection_name = collection_name or store_cfg.get("collection_name", "documents")
        distance_metric = distance_metric or store_cfg.get("distance_metric", "cosine")

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False),
        )

        self.collection_name = collection_name
        self.distance_metric = distance_metric
        self.collection = None

    def get_or_create_collection(
        self,
        embedding_dimension: Optional[int] = None,
    ):
        """Get or create the collection."""
        metadata = None
        if embedding_dimension:
            metadata = {"embedding_dimension": embedding_dimension}

        # Without an explicit space Chroma uses squared L2, which made
        # "score = 1 - distance" meaningless. The space is fixed at creation.
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata=metadata,
            configuration={"hnsw": {"space": self.distance_metric}},
        )

        space = (self.collection.configuration_json or {}).get("hnsw", {}).get("space")
        if space and space != self.distance_metric:
            print(
                f"Warning: collection '{self.collection_name}' uses '{space}' distance, "
                f"config asks for '{self.distance_metric}'. Re-run ingest with --reset."
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
            ids.append(chunk_id)
            embeddings.append(chunk["embedding"])
            documents.append(chunk.get("contextualized_text", chunk["text"]))

            metadata = {
                "document_id": document_id,
                "text": chunk["text"][:1000],
                "content_type": chunk.get("content_type", "text"),
                "document_title": chunk.get("document_title", document_id),
            }

            if chunk.get("page_start") is not None:
                metadata["page_start"] = chunk["page_start"]
            if chunk.get("page_end") is not None:
                metadata["page_end"] = chunk["page_end"]
            if chunk.get("headings"):
                metadata["headings"] = " > ".join(chunk["headings"])

            metadatas.append(metadata)

        # Chunk ids are positional, so a re-chunked document with fewer chunks
        # would leave its old tail chunks behind. Drop ids no longer produced.
        existing = self.collection.get(where={"document_id": document_id}, include=[])["ids"]
        stale = sorted(set(existing) - set(ids))
        if stale:
            self.collection.delete(ids=stale)
            print(f"  -> removed {len(stale)} stale chunks of {document_id}")

        # Index in batches
        indexed = 0
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_documents = documents[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]

            # upsert, not add-if-missing: re-chunked or re-embedded chunks
            # must replace the stale vectors stored under the same id.
            self.collection.upsert(
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

    def get_all(self) -> Dict[str, Any]:
        """Return every stored chunk id, document and metadata."""
        if self.collection is None:
            self.get_or_create_collection()
        return self.collection.get(include=["documents", "metadatas"])

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
        model_name: Optional[str] = None,
        collection_name: Optional[str] = None,
        hybrid: Optional[bool] = None,
        rerank: Optional[bool] = None,
    ):
        """
        Initialize the retriever.

        Args:
            vector_store_dir: Path to vector store directory.
            model_name: Embedding model name (default: retriever_config.yaml).
            collection_name: Collection name (default: retriever_config.yaml).
            hybrid: Fuse BM25 with dense search (default: retriever_config.yaml).
            rerank: Rerank with a cross-encoder (default: retriever_config.yaml).
        """
        config = load_retriever_config()
        embedding_cfg = config.get("embedding", {})
        retrieval_cfg = config.get("retrieval", {})
        hybrid_cfg = retrieval_cfg.get("hybrid", {})
        rerank_cfg = config.get("reranking", {})

        model_name = model_name or embedding_cfg.get("model_name", "all-MiniLM-L6-v2")
        self.normalize = embedding_cfg.get("normalize", True)
        self.default_top_k = retrieval_cfg.get("top_k", 5)
        self.candidate_k = retrieval_cfg.get("candidate_k", 20)
        self.rrf_k = hybrid_cfg.get("rrf_k", 60)

        self.vector_store = VectorStore(
            persist_dir=vector_store_dir,
            collection_name=collection_name,
        )
        self.vector_store.get_or_create_collection()

        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"Model loaded. Dimension: {self.model.get_embedding_dimension()}")

        self.hybrid = hybrid_cfg.get("enabled", False) if hybrid is None else hybrid
        self.bm25 = None
        self._chunks_by_id: Dict[str, Dict[str, Any]] = {}
        if self.hybrid:
            stored = self.vector_store.get_all()
            for chunk_id, document, metadata in zip(
                stored["ids"], stored["documents"], stored["metadatas"]
            ):
                self._chunks_by_id[chunk_id] = {"document": document, "metadata": metadata}
            self.bm25 = BM25Index(
                stored["ids"],
                stored["documents"],
                k1=hybrid_cfg.get("bm25_k1", 1.5),
                b=hybrid_cfg.get("bm25_b", 0.75),
            )
            print(f"BM25 index built over {len(stored['ids'])} chunks.")

        use_rerank = rerank_cfg.get("enabled", False) if rerank is None else rerank
        self.reranker = None
        if use_rerank:
            from src.reranking import CrossEncoderReranker
            self.reranker = CrossEncoderReranker(
                rerank_cfg.get("model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            )

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        document_id: Optional[str] = None,
        content_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks.

        Dense search (+ BM25 fused with RRF when hybrid is on) gathers
        candidate_k candidates; the cross-encoder, when on, picks the top_k.

        Args:
            query: Search query.
            top_k: Number of results (default: retriever_config.yaml).
            document_id: Filter by document.
            content_type: Filter by content type.

        Returns:
            list: Search results, best first.
        """
        top_k = top_k or self.default_top_k

        # Build filter (Chroma needs $and for more than one condition)
        conditions = []
        if document_id:
            conditions.append({"document_id": document_id})
        if content_type:
            conditions.append({"content_type": content_type})
        where = None
        if len(conditions) == 1:
            where = conditions[0]
        elif conditions:
            where = {"$and": conditions}

        first_stage_k = top_k
        if self.bm25 is not None or self.reranker is not None:
            first_stage_k = max(top_k, self.candidate_k)

        query_embedding = self.model.encode(
            query, normalize_embeddings=self.normalize
        ).tolist()

        dense = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=first_stage_k,
            where=where,
        )

        candidates = dense
        if self.bm25 is not None:
            by_id = {r["chunk_id"]: r for r in dense}
            sparse = [
                chunk_id
                for chunk_id, _ in self.bm25.search(query, top_k=first_stage_k)
                if self._matches(chunk_id, document_id, content_type)
            ]
            fused = reciprocal_rank_fusion(
                [[r["chunk_id"] for r in dense], sparse], k=self.rrf_k
            )[:first_stage_k]

            candidates = []
            for chunk_id, rrf_score in fused:
                result = by_id.get(chunk_id)
                if result is None:
                    stored = self._chunks_by_id[chunk_id]
                    result = {
                        "chunk_id": chunk_id,
                        "document": stored["document"],
                        "metadata": stored["metadata"],
                        "distance": None,
                        "score": None,
                    }
                candidates.append({**result, "rrf_score": rrf_score})

        if self.reranker is not None:
            return self.reranker.rerank(query, candidates, top_k=top_k)
        return candidates[:top_k]

    def _matches(
        self,
        chunk_id: str,
        document_id: Optional[str],
        content_type: Optional[str],
    ) -> bool:
        metadata = self._chunks_by_id[chunk_id]["metadata"]
        if document_id and metadata.get("document_id") != document_id:
            return False
        if content_type and metadata.get("content_type") != content_type:
            return False
        return True

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
                if result.get("rerank_score") is not None:
                    lines.append(f"Rerank score: {result['rerank_score']:.4f}")
                if result.get("score") is not None:
                    lines.append(f"Cosine similarity: {result['score']:.4f}")

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
