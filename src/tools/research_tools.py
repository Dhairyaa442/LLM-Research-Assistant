"""
Agent tools: web search, arXiv lookup, Wikipedia, and RAG retrieval.
Each tool is a LangChain-compatible callable used by the agent graph.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import arxiv
import wikipedia
from langchain.schema import Document
from langchain_core.tools import tool

if TYPE_CHECKING:
    from src.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


def make_rag_tool(rag: "RAGPipeline"):
    """Factory that creates a RAG retrieval tool bound to a pipeline instance."""

    @tool
    def rag_retriever(query: str) -> str:
        """
        Search the local knowledge base using semantic + keyword hybrid retrieval.
        Use this FIRST for any question that may be answerable from ingested documents.

        Args:
            query: The research question or topic to search for.

        Returns:
            Relevant document excerpts with source metadata.
        """
        docs = rag.retrieve(query, k=6)
        if not docs:
            return "No relevant documents found in the knowledge base."

        results = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Unknown")
            results.append(f"[{i}] Source: {source}\n{doc.page_content}\n")
        return "\n---\n".join(results)

    return rag_retriever


@tool
def arxiv_search(query: str) -> str:
    """
    Search arXiv for recent academic papers on a topic.
    Use this for finding cutting-edge research, technical details, and citations.

    Args:
        query: Research topic or paper title to search for.

    Returns:
        List of papers with titles, authors, abstracts, and arXiv IDs.
    """
    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=4,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        papers = list(client.results(search))
        if not papers:
            return f"No arXiv papers found for: {query}"

        results = []
        for paper in papers:
            authors = ", ".join(str(a) for a in paper.authors[:3])
            if len(paper.authors) > 3:
                authors += " et al."
            results.append(
                f"Title: {paper.title}\n"
                f"Authors: {authors}\n"
                f"Published: {paper.published.strftime('%Y-%m-%d')}\n"
                f"arXiv ID: {paper.entry_id.split('/')[-1]}\n"
                f"Abstract: {paper.summary[:400]}...\n"
            )
        return "\n---\n".join(results)
    except Exception as exc:
        logger.error("arXiv search error: %s", exc)
        return f"arXiv search failed: {exc}"


@tool
def wikipedia_search(query: str) -> str:
    """
    Look up a topic on Wikipedia for background knowledge and definitions.
    Use this for foundational concepts, historical context, or general overviews.

    Args:
        query: Topic to look up on Wikipedia.

    Returns:
        Wikipedia summary (first ~800 characters).
    """
    try:
        wikipedia.set_lang("en")
        page = wikipedia.page(query, auto_suggest=True)
        summary = wikipedia.summary(query, sentences=6, auto_suggest=True)
        return f"**{page.title}**\n\n{summary}\n\nSource: {page.url}"
    except wikipedia.exceptions.DisambiguationError as e:
        # Try the first option
        try:
            summary = wikipedia.summary(e.options[0], sentences=6)
            return f"**{e.options[0]}** (disambiguation resolved)\n\n{summary}"
        except Exception:
            return f"Disambiguation: {query} could refer to: {', '.join(e.options[:5])}"
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for: {query}"
    except Exception as exc:
        logger.error("Wikipedia error: %s", exc)
        return f"Wikipedia lookup failed: {exc}"


@tool
def citation_formatter(raw_text: str) -> str:
    """
    Format a list of sources or findings into a clean, numbered citation list.
    Use this as the FINAL step before producing your answer to format references.

    Args:
        raw_text: Unformatted source information or bibliography notes.

    Returns:
        Numbered, formatted citation list.
    """
    lines = [line.strip() for line in raw_text.strip().split("\n") if line.strip()]
    formatted = []
    for i, line in enumerate(lines, 1):
        formatted.append(f"[{i}] {line}")
    return "\n".join(formatted) if formatted else "No citations provided."
