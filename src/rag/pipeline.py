"""
RAG Pipeline: document ingestion, chunking, embedding, and retrieval.

Uses FAISS for vector storage and OpenAI text-embedding-3-small for embeddings.
Supports hybrid retrieval: dense (semantic) + BM25 (keyword).
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import List, Optional

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    OPENAI_API_KEY,
)

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Manages the full RAG lifecycle:
      1. Ingest raw documents → chunk → embed → store in FAISS
      2. Hybrid retrieval (dense FAISS + sparse BM25) with RRF fusion
    """

    def __init__(self, index_path: Optional[Path] = None) -> None:
        self.index_path = index_path or FAISS_INDEX_PATH
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.vector_store: Optional[FAISS] = None
        self.bm25_retriever: Optional[BM25Retriever] = None
        self._all_chunks: List[Document] = []

        # Attempt to load existing index
        if self._index_exists():
            self._load_index()

    # ─── Ingestion ────────────────────────────────────────────────────────────

    def ingest(self, documents: List[Document]) -> int:
        """
        Chunk, embed, and index a list of Documents.

        Args:
            documents: Raw LangChain Document objects.

        Returns:
            Number of chunks indexed.
        """
        chunks = self.splitter.split_documents(documents)
        logger.info("Split %d documents into %d chunks", len(documents), len(chunks))

        if not chunks:
            logger.warning("No chunks produced — skipping ingestion.")
            return 0

        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vector_store.add_documents(chunks)

        self._all_chunks.extend(chunks)
        self.bm25_retriever = BM25Retriever.from_documents(self._all_chunks)
        self.bm25_retriever.k = 5

        self._save_index()
        logger.info("Indexed %d chunks successfully.", len(chunks))
        return len(chunks)

    # ─── Retrieval ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = 6) -> List[Document]:
        """
        Hybrid retrieval: merge FAISS dense results + BM25 sparse results
        using Reciprocal Rank Fusion (RRF).

        Args:
            query: The search query string.
            k: Number of final documents to return.

        Returns:
            Ranked list of relevant Documents.
        """
        if self.vector_store is None:
            logger.warning("Vector store is empty — no documents ingested yet.")
            return []

        # Dense retrieval
        dense_docs = self.vector_store.similarity_search(query, k=k)

        # Sparse retrieval
        sparse_docs: List[Document] = []
        if self.bm25_retriever:
            self.bm25_retriever.k = k
            sparse_docs = self.bm25_retriever.invoke(query)

        # RRF fusion
        fused = self._reciprocal_rank_fusion(dense_docs, sparse_docs, k=k)
        logger.debug("Retrieved %d docs for query: '%s'", len(fused), query[:60])
        return fused

    @staticmethod
    def _reciprocal_rank_fusion(
        list_a: List[Document],
        list_b: List[Document],
        k: int = 6,
        rrf_k: int = 60,
    ) -> List[Document]:
        """Merge two ranked lists via RRF scoring."""
        scores: dict[str, float] = {}
        doc_map: dict[str, Document] = {}

        for rank, doc in enumerate(list_a):
            key = doc.page_content[:200]
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            doc_map[key] = doc

        for rank, doc in enumerate(list_b):
            key = doc.page_content[:200]
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            doc_map[key] = doc

        ranked_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
        return [doc_map[key] for key in ranked_keys[:k]]

    # ─── Persistence ──────────────────────────────────────────────────────────

    def _save_index(self) -> None:
        self.index_path.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(str(self.index_path))
        with open(self.index_path / "chunks.pkl", "wb") as f:
            pickle.dump(self._all_chunks, f)
        logger.info("Saved FAISS index to %s", self.index_path)

    def _load_index(self) -> None:
        try:
            self.vector_store = FAISS.load_local(
                str(self.index_path),
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            chunks_path = self.index_path / "chunks.pkl"
            if chunks_path.exists():
                with open(chunks_path, "rb") as f:
                    self._all_chunks = pickle.load(f)
                self.bm25_retriever = BM25Retriever.from_documents(self._all_chunks)
                self.bm25_retriever.k = 5
            logger.info("Loaded existing FAISS index (%d chunks).", len(self._all_chunks))
        except Exception as exc:
            logger.error("Failed to load index: %s — starting fresh.", exc)
            self.vector_store = None

    def _index_exists(self) -> bool:
        return (self.index_path / "index.faiss").exists()
