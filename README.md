# 🔬 LLM Research Assistant

> A production-grade **multi-agent research pipeline** powered by GPT-4o, LangGraph, and Hybrid RAG (FAISS + BM25).

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2-green)](https://github.com/langchain-ai/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## 🧠 What It Does

Given any research question, this system spins up a **4-agent pipeline** that:

1. **Plans** — decomposes your query into focused sub-questions
2. **Researches** — autonomously calls RAG retrieval, arXiv, and Wikipedia in a tool-use loop
3. **Critiques** — self-evaluates for hallucinations and gaps, loops back if needed
4. **Synthesizes** — produces a structured answer with inline citations

---

## 🏗️ Architecture

```
User Query
    │
    ▼
┌─────────────┐
│   Planner   │  GPT-4o decomposes query → 3-5 sub-questions
└──────┬──────┘
       │
┌──────▼──────────────────────────────────────────────────┐
│                    Researcher Agent                      │
│  ┌────────────┐  ┌──────────────┐  ┌─────────────────┐  │
│  │ RAG (FAISS │  │ arXiv Search │  │ Wikipedia Search│  │
│  │  + BM25)   │  │              │  │                 │  │
│  └────────────┘  └──────────────┘  └─────────────────┘  │
│         ↑ Tool calls loop until sufficient evidence      │
└──────┬──────────────────────────────────────────────────┘
       │
┌──────▼──────┐
│   Critic    │  Detects hallucinations/gaps → loops back if NEEDS_REVISION
└──────┬──────┘
       │
┌──────▼──────┐
│ Synthesizer │  Writes final structured answer with [1][2][3] citations
└─────────────┘
```

### Key Technical Highlights

| Feature | Implementation |
|---|---|
| **Orchestration** | LangGraph `StateGraph` with typed state and conditional routing |
| **Dense Retrieval** | FAISS with OpenAI `text-embedding-3-small` |
| **Sparse Retrieval** | BM25 (keyword matching) |
| **Fusion** | Reciprocal Rank Fusion (RRF) merging dense + sparse results |
| **Reflection Loop** | Critic agent routes back to researcher if evidence is insufficient |
| **Persistence** | FAISS index saved to disk; reloaded on startup |
| **API** | FastAPI with Pydantic v2 schemas |
| **UI** | Streamlit with live agent step streaming |
| **Deployment** | Docker + docker-compose ready |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/dhairya0402/llm-research-assistant.git
cd llm-research-assistant
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Run the UI

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) and start researching.

### 4. Run the API

```bash
uvicorn api:app --reload --port 8000
```

API docs at [http://localhost:8000/docs](http://localhost:8000/docs)

### 5. Docker (optional)

```bash
docker-compose up --build
```

---

## 💡 Example Queries

```
"What are the key differences between RAG and fine-tuning for LLM knowledge injection?"

"Summarize recent advances in LLM agent architectures and their limitations."

"How does RLHF compare to DPO for aligning language models?"
```

---

## 📡 API Reference

### `POST /research`

```json
{
  "query": "What is retrieval-augmented generation?"
}
```

**Response:**
```json
{
  "query": "What is retrieval-augmented generation?",
  "answer": "## Retrieval-Augmented Generation (RAG)\n\nRAG is a technique...",
  "sub_questions": ["What is RAG?", "How does RAG differ from fine-tuning?", "..."],
  "iterations": 2,
  "elapsed_seconds": 14.3
}
```

### `POST /ingest`

```json
{
  "documents": [
    { "content": "LangChain is a framework...", "source": "langchain_docs.txt" }
  ]
}
```

---

## 🧪 Tests

```bash
pytest tests/ -v
```

Tests cover:
- RAG chunking and RRF fusion logic
- Tool output formatting
- Graph compilation smoke test

---

## 📁 Project Structure

```
llm-research-assistant/
├── src/
│   ├── agents/
│   │   └── graph.py          # LangGraph multi-agent graph
│   ├── rag/
│   │   └── pipeline.py       # FAISS + BM25 hybrid RAG
│   ├── tools/
│   │   └── research_tools.py # arXiv, Wikipedia, RAG, citation tools
│   └── config.py             # Centralized settings
├── tests/
│   └── test_rag_and_tools.py
├── app.py                    # Streamlit UI
├── api.py                    # FastAPI backend
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## 🔧 Configuration

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | Required. Your OpenAI API key. |
| `MODEL_NAME` | `gpt-4o` | LLM model to use |
| `MAX_ITERATIONS` | `10` | Max researcher→critic loops |
| `CHUNK_SIZE` | `512` | Token chunk size for splitting |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `FAISS_INDEX_PATH` | `./data/faiss_index` | Where to persist the vector index |

---

## 📄 License

MIT — feel free to use, modify, and build on this.
