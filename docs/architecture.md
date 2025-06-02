# Architecture Overview

## System Design

The LLM Research Assistant is a multi-agent pipeline built on LangGraph. Each agent is a node in a directed state graph; edges are either unconditional or routing functions that inspect the current state.

---

## Agent Roles

### 1. Planner
- **Input:** Raw user query
- **Output:** 3–5 focused sub-questions
- **Model:** GPT-4o (zero-temperature)
- **Why it matters:** Decomposing the query prevents the researcher from drifting off-topic and ensures all facets of the question are covered.

### 2. Researcher
- **Input:** Sub-questions + conversation history
- **Output:** Research notes, tool call results
- **Tools available:**
  - `rag_retriever` — local FAISS+BM25 knowledge base
  - `arxiv_search` — live academic paper search
  - `wikipedia_search` — background and definitions
  - `citation_formatter` — formats references
- **Loop:** The researcher node re-runs after each tool call until the LLM stops calling tools, then hands off to the Critic.

### 3. Critic
- **Input:** All research notes + original query
- **Output:** `APPROVED` or `NEEDS_REVISION: <specific gaps>`
- **Routing:** If `NEEDS_REVISION` and under `MAX_ITERATIONS`, routes back to Researcher. Otherwise proceeds to Synthesizer.
- **Why it matters:** Self-critique is the primary mechanism for reducing hallucinations and ensuring completeness.

### 4. Synthesizer
- **Input:** All research notes
- **Output:** Final structured answer with inline citations and a References section
- **Model:** GPT-4o

---

## RAG Pipeline

```
Document(s)
    │
    ▼
RecursiveCharacterTextSplitter (chunk_size=512, overlap=64)
    │
    ├──► OpenAI text-embedding-3-small ──► FAISS index (dense)
    │
    └──► BM25Retriever (sparse / keyword)

At query time:
    query ──► FAISS top-k  ─┐
    query ──► BM25 top-k   ─┤──► RRF Fusion ──► top-k merged docs
```

**Reciprocal Rank Fusion (RRF):**  
Each document gets a score of `1 / (rrf_k + rank)` from each list. Scores are summed and documents re-ranked. This reliably outperforms either retriever alone, especially on out-of-domain queries.

---

## State Schema

```python
class AgentState(TypedDict):
    messages: List[BaseMessage]   # Full conversation history
    query: str                    # Original user query
    sub_questions: List[str]      # Planner output
    research_notes: List[str]     # Researcher accumulations
    critique: str                 # Critic verdict
    final_answer: str             # Synthesizer output
    iterations: int               # Loop counter (bounded by MAX_ITERATIONS)
```

---

## Graph Topology

```
START
  └─► planner
        └─► researcher
              ├─► tools (if tool_calls present)  ──► researcher (loop)
              └─► critic
                    ├─► researcher (if NEEDS_REVISION and iterations < MAX)
                    └─► synthesizer
                          └─► END
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI GPT-4o |
| Agent Orchestration | LangGraph 0.2 |
| Embeddings | OpenAI text-embedding-3-small |
| Dense Vector Store | FAISS (faiss-cpu) |
| Sparse Retrieval | BM25 (langchain-community) |
| Fusion | Reciprocal Rank Fusion (custom) |
| External Tools | arXiv API, Wikipedia API |
| API Server | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Containerization | Docker + docker-compose |
| Testing | pytest + unittest.mock |
