"""
Unit tests for the RAG pipeline and tools.
Run with: pytest tests/ -v
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain.schema import Document

from src.rag.pipeline import RAGPipeline


# ─── RAG Pipeline Tests ───────────────────────────────────────────────────────

class TestRAGPipeline:
    """Tests for document ingestion and retrieval."""

    @pytest.fixture
    def mock_rag(self, tmp_path):
        """RAG pipeline with mocked embeddings to avoid API calls."""
        with patch("src.rag.pipeline.OpenAIEmbeddings") as mock_emb:
            # Return a deterministic fake embedding vector
            mock_emb.return_value.embed_documents.return_value = [[0.1] * 1536]
            mock_emb.return_value.embed_query.return_value = [0.1] * 1536
            rag = RAGPipeline(index_path=tmp_path / "test_index")
            yield rag

    def test_ingest_returns_chunk_count(self, mock_rag):
        """Ingesting documents should return a positive chunk count."""
        docs = [
            Document(page_content="LangChain is a framework for building LLM applications." * 5,
                     metadata={"source": "test.txt"}),
        ]
        with patch.object(mock_rag.embeddings, "embed_documents", return_value=[[0.1] * 1536]):
            # We mock FAISS.from_documents to avoid actual embedding calls
            with patch("src.rag.pipeline.FAISS") as mock_faiss:
                mock_faiss.from_documents.return_value = MagicMock()
                count = mock_rag.ingest(docs)
        assert count >= 1

    def test_retrieve_returns_empty_when_no_index(self, tmp_path):
        """Retrieval on an empty index should return an empty list gracefully."""
        with patch("src.rag.pipeline.OpenAIEmbeddings"):
            rag = RAGPipeline(index_path=tmp_path / "empty_index")
        results = rag.retrieve("what is RAG?")
        assert results == []

    def test_rrf_fusion_deduplicates(self):
        """RRF fusion should deduplicate documents appearing in both lists."""
        doc_a = Document(page_content="shared content", metadata={})
        doc_b = Document(page_content="unique content A", metadata={})
        doc_c = Document(page_content="unique content B", metadata={})

        fused = RAGPipeline._reciprocal_rank_fusion([doc_a, doc_b], [doc_a, doc_c], k=10)
        contents = [d.page_content for d in fused]
        assert contents.count("shared content") == 1, "Deduplication failed"

    def test_rrf_fusion_respects_k(self):
        """RRF fusion should return at most k documents."""
        docs_a = [Document(page_content=f"doc {i}", metadata={}) for i in range(10)]
        docs_b = [Document(page_content=f"doc {i+10}", metadata={}) for i in range(10)]
        fused = RAGPipeline._reciprocal_rank_fusion(docs_a, docs_b, k=4)
        assert len(fused) <= 4

    def test_rrf_empty_lists(self):
        """RRF should handle empty input lists without error."""
        result = RAGPipeline._reciprocal_rank_fusion([], [], k=5)
        assert result == []


# ─── Tool Tests ───────────────────────────────────────────────────────────────

class TestTools:
    """Tests for research tools with mocked external calls."""

    def test_arxiv_search_returns_string(self):
        """arXiv search result should be a non-empty string."""
        with patch("src.tools.research_tools.arxiv") as mock_arxiv:
            mock_paper = MagicMock()
            mock_paper.title = "Test Paper on RAG"
            mock_paper.authors = [MagicMock(__str__=lambda s: "Author A")]
            mock_paper.published.strftime.return_value = "2024-01-01"
            mock_paper.entry_id = "https://arxiv.org/abs/2401.00001"
            mock_paper.summary = "This paper discusses RAG systems." * 5
            mock_arxiv.Client.return_value.results.return_value = [mock_paper]
            mock_arxiv.Search = MagicMock()
            mock_arxiv.SortCriterion.Relevance = "relevance"

            from src.tools.research_tools import arxiv_search
            result = arxiv_search.invoke("retrieval augmented generation")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_citation_formatter_numbers_lines(self):
        """Citation formatter should number each line correctly."""
        from src.tools.research_tools import citation_formatter
        raw = "Smith et al., 2023\nJones & Lee, 2024"
        result = citation_formatter.invoke(raw)
        assert "[1]" in result
        assert "[2]" in result

    def test_citation_formatter_handles_empty(self):
        """Citation formatter should handle empty input gracefully."""
        from src.tools.research_tools import citation_formatter
        result = citation_formatter.invoke("   ")
        assert "No citations" in result


# ─── Integration Smoke Test ───────────────────────────────────────────────────

class TestGraphIntegration:
    """Smoke test the graph builds without API calls."""

    def test_graph_compiles(self, tmp_path):
        """Graph should compile without errors given a valid RAG pipeline."""
        with patch("src.rag.pipeline.OpenAIEmbeddings"):
            rag = RAGPipeline(index_path=tmp_path / "smoke_index")
        with patch("src.agents.graph.ChatOpenAI"):
            from src.agents.graph import build_research_graph
            graph = build_research_graph(rag)
        assert graph is not None
