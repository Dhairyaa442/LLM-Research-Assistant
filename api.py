"""
FastAPI backend for the LLM Research Assistant.

Endpoints:
  POST /research   — run the full multi-agent pipeline
  POST /ingest     — add documents to the RAG knowledge base
  GET  /health     — liveness check

Run with:
    uvicorn api:app --reload --port 8000
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from langchain.schema import Document
from pydantic import BaseModel, Field

from src.agents.graph import build_research_graph
from src.rag.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Research Assistant API",
    description="Multi-agent research pipeline with RAG, arXiv, and Wikipedia.",
    version="1.0.0",
)

# Shared singletons
_rag: Optional[RAGPipeline] = None
_graph = None


def get_rag() -> RAGPipeline:
    global _rag
    if _rag is None:
        _rag = RAGPipeline()
    return _rag


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_research_graph(get_rag())
    return _graph


# ─── Schemas ──────────────────────────────────────────────────────────────────

class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=5, max_length=1000, description="Research question")


class ResearchResponse(BaseModel):
    query: str
    answer: str
    sub_questions: List[str]
    iterations: int
    elapsed_seconds: float


class IngestRequest(BaseModel):
    documents: List[dict] = Field(
        ..., description="List of {content: str, source: str} dicts"
    )


class IngestResponse(BaseModel):
    chunks_indexed: int
    documents_received: int


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "chunks_in_index": len(get_rag()._all_chunks)}


@app.post("/research", response_model=ResearchResponse)
def research(req: ResearchRequest):
    """Run the full multi-agent research pipeline for a given query."""
    logger.info("Research request: %s", req.query[:80])
    start = time.time()

    initial_state = {
        "messages": [],
        "query": req.query,
        "sub_questions": [],
        "research_notes": [],
        "critique": "",
        "final_answer": "",
        "iterations": 0,
    }

    try:
        final_state = get_graph().invoke(initial_state)
    except Exception as exc:
        logger.exception("Graph error")
        raise HTTPException(status_code=500, detail=str(exc))

    return ResearchResponse(
        query=req.query,
        answer=final_state.get("final_answer", ""),
        sub_questions=final_state.get("sub_questions", []),
        iterations=final_state.get("iterations", 0),
        elapsed_seconds=round(time.time() - start, 2),
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    """Ingest documents into the RAG knowledge base."""
    docs = [
        Document(
            page_content=d.get("content", ""),
            metadata={"source": d.get("source", "unknown")},
        )
        for d in req.documents
        if d.get("content")
    ]
    if not docs:
        raise HTTPException(status_code=400, detail="No valid document content provided.")

    chunks = get_rag().ingest(docs)
    return IngestResponse(chunks_indexed=chunks, documents_received=len(docs))
