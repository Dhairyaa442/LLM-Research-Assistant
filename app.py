import logging
import time

import streamlit as st
from langchain.schema import Document

from src.agents.graph import build_research_graph
from src.rag.pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="LLM Research Assistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.agent-step { background: #f0f4ff; border-left: 4px solid #4a6cf7;
              padding: 0.6rem 1rem; border-radius: 4px; margin: 0.3rem 0; font-size: 0.9rem; }
.answer-box { background: #fafafa; border: 1px solid #e0e0e0;
              border-radius: 8px; padding: 1.5rem; margin-top: 1rem; color: #1a1a1a; }
</style>
""", unsafe_allow_html=True)

if "rag" not in st.session_state:
    st.session_state.rag = RAGPipeline()
if "graph" not in st.session_state:
    st.session_state.graph = build_research_graph(st.session_state.rag)
if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Knowledge Base")
    st.caption("Upload docs to give the agent extra context via RAG.")

    uploaded_files = st.file_uploader(
        "Upload PDFs or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Ingest Documents", type="primary"):
        docs = []
        for f in uploaded_files:
            content = f.read().decode("utf-8", errors="ignore")
            docs.append(Document(page_content=content, metadata={"source": f.name}))
        with st.spinner("Embedding and indexing..."):
            n = st.session_state.rag.ingest(docs)
        st.success(f"Indexed {n} chunks from {len(uploaded_files)} file(s).")

    st.divider()
    st.header("Settings")
    show_steps = st.toggle("Show agent reasoning steps", value=True)
    st.caption(f"Index: {st.session_state.rag.index_path}")
    st.caption(f"Chunks in index: {len(st.session_state.rag._all_chunks)}")

st.title("🔬 LLM Research Assistant")
st.caption("Multi-agent AI powered by GPT-4o · LangGraph · RAG (FAISS + BM25)")

for entry in st.session_state.history:
    with st.chat_message("user"):
        st.write(entry["query"])
    with st.chat_message("assistant"):
        st.markdown(entry["answer"])

query = st.chat_input("Ask a research question...")

if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        steps_container = st.empty()
        answer_placeholder = st.empty()
        step_log = []

        def log_step(icon: str, text: str):
            step_log.append(f'<div class="agent-step">{icon} {text}</div>')
            if show_steps:
                steps_container.markdown("\n".join(step_log), unsafe_allow_html=True)

        log_step("🧭", "Planner: breaking down your query...")
        start = time.time()

        try:
            initial_state = {
                "messages": [],
                "query": query,
                "sub_questions": [],
                "research_notes": [],
                "critique": "",
                "final_answer": "",
                "iterations": 0,
            }

            icons = {
                "planner": "🧭", "researcher": "🔍", "tools": "🛠️",
                "extract": "📋", "critic": "🧐", "synthesizer": "✍️",
            }
            labels = {
                "planner": "Planner: building research plan",
                "researcher": "Researcher: gathering information",
                "tools": "Tools: running searches",
                "extract": "Extracting tool results",
                "critic": "Critic: reviewing quality",
                "synthesizer": "Synthesizer: writing answer",
            }

            all_states = {}
            final_state = None

            for step in st.session_state.graph.stream(initial_state):
                node_name = list(step.keys())[0]
                if node_name in labels:
                    log_step(icons.get(node_name, "⚙️"), labels[node_name])
                all_states.update(step.get(node_name, {}))
                if node_name == "synthesizer":
                    final_state = step.get(node_name, {})

            if final_state is None:
                final_state = all_states

            elapsed = time.time() - start
            log_step("✅", f"Done in {elapsed:.1f}s")

            answer = final_state.get("final_answer", "No answer generated.")
            steps_container.markdown("\n".join(step_log), unsafe_allow_html=True)
            answer_placeholder.markdown(
                f'<div class="answer-box">{answer}</div>', unsafe_allow_html=True
            )
            st.session_state.history.append({"query": query, "answer": answer})

        except Exception as exc:
            st.error(f"Error: {exc}")
            logging.exception("Graph execution error")