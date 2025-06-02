"""
Multi-agent research pipeline using LangGraph.

Four nodes: planner breaks the query apart, researcher hits tools,
extract pulls tool outputs into state, critic checks quality,
synthesizer writes the final answer.

Took a while to get the tool message handling right - the key insight
is that tool results live in messages but need to be explicitly pulled
into research_notes before the critic sees them.
"""

from __future__ import annotations

import logging
from typing import Annotated, List, Literal, Sequence, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.config import MAX_ITERATIONS, MODEL_NAME, OPENAI_API_KEY, TEMPERATURE
from src.rag.pipeline import RAGPipeline
from src.tools.research_tools import (
    arxiv_search,
    citation_formatter,
    make_rag_tool,
    wikipedia_search,
)

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query: str
    sub_questions: List[str]
    research_notes: List[str]
    critique: str
    final_answer: str
    iterations: int


PLANNER_SYSTEM = """You are a research planning expert. Given a user's research query,
decompose it into 3-5 focused sub-questions that together would fully answer the query.
Return ONLY a numbered list of sub-questions, nothing else.

Example:
1. What is the historical background of X?
2. What are the current state-of-the-art approaches to X?
3. What are the limitations of existing work on X?
"""

RESEARCHER_SYSTEM = """You are an expert research agent. Answer the sub-questions using tools.

IMPORTANT: You MUST call at least 2 tools. Always call arxiv_search AND wikipedia_search.
The rag_retriever may be empty - if it returns nothing, that is fine, just move on.

Required steps:
1. Call arxiv_search with the main topic
2. Call wikipedia_search with the main topic
3. Summarize what you found
"""

CRITIC_SYSTEM = """You are a research critic. Review the research notes.

If the notes contain actual content from arxiv or wikipedia, respond with: APPROVED
If the notes are completely empty, respond with: APPROVED (no sources available)

Be brief.
"""

SYNTHESIZER_SYSTEM = """You are an expert writer. Synthesize the research notes
into a comprehensive, well-structured answer.

Requirements:
- Start with a clear, direct answer to the query
- Organize with headers (## Section Name) where appropriate
- Cite sources inline as [1], [2], etc.
- Include a ## References section at the end
- Be accurate - only include information supported by the research notes
- Length: 300-800 words depending on complexity
"""


def build_research_graph(rag_pipeline: RAGPipeline):
    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        openai_api_key=OPENAI_API_KEY,
    )

    rag_tool = make_rag_tool(rag_pipeline)
    tools = [rag_tool, arxiv_search, wikipedia_search, citation_formatter]
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    def planner_node(state: AgentState) -> AgentState:
        logger.info("[Planner] Decomposing query: %s", state["query"][:80])
        response = llm.invoke([
            SystemMessage(content=PLANNER_SYSTEM),
            HumanMessage(content=f"Research query: {state['query']}"),
        ])
        lines = [
            line.strip().lstrip("0123456789.- ").strip()
            for line in response.content.split("\n")
            if line.strip() and line.strip()[0].isdigit()
        ]
        sub_questions = lines if lines else [state["query"]]
        logger.info("[Planner] Generated %d sub-questions.", len(sub_questions))
        return {
            **state,
            "sub_questions": sub_questions,
            "messages": state["messages"] + [response],
        }

    def researcher_node(state: AgentState) -> AgentState:
        logger.info("[Researcher] Iteration %d", state.get("iterations", 0) + 1)
        sub_qs = "\n".join(f"{i+1}. {q}" for i, q in enumerate(state["sub_questions"]))
        prompt = (
            f"Original query: {state['query']}\n\n"
            f"Sub-questions to answer:\n{sub_qs}\n\n"
            "Use the available tools to research each sub-question. "
            "Gather evidence from multiple sources. Be thorough."
        )

        
        messages = [
            SystemMessage(content=RESEARCHER_SYSTEM),
            HumanMessage(content=prompt),
        ]

        response = llm_with_tools.invoke(messages)
        notes = state.get("research_notes", [])
        if response.content:
            notes = notes + [response.content]

        return {
            **state,
            "messages": state["messages"] + [response],
            "research_notes": notes,
            "iterations": state.get("iterations", 0) + 1,
        }

    def extract_tool_results(state: AgentState) -> AgentState:
        # the tool results land in messages but the critic reads them through research_notes, so we pull them across
        notes = list(state.get("research_notes", []))
        for msg in state["messages"]:
            if hasattr(msg, "type") and msg.type == "tool" and msg.content:
                tool_name = getattr(msg, "name", "tool")
                notes.append(f"[Tool: {tool_name}]\n{msg.content}")
            elif type(msg).__name__ == "ToolMessage" and msg.content:
                tool_name = getattr(msg, "name", "tool")
                notes.append(f"[Tool: {tool_name}]\n{msg.content}")
        logger.info("[Extract] Captured %d research notes.", len(notes))
        return {**state, "research_notes": notes}

    def critic_node(state: AgentState) -> AgentState:
        logger.info("[Critic] Reviewing research notes.")
        notes_text = "\n\n---\n\n".join(state.get("research_notes", []))
        response = llm.invoke([
            SystemMessage(content=CRITIC_SYSTEM),
            HumanMessage(
                content=(
                    f"Original query: {state['query']}\n\n"
                    f"Research notes:\n{notes_text[:3000]}"
                )
            ),
        ])
        logger.info("[Critic] Verdict: %s", response.content[:100])
        return {
            **state,
            "critique": response.content,
            "messages": state["messages"] + [response],
        }

    def synthesizer_node(state: AgentState) -> AgentState:
        logger.info("[Synthesizer] Writing final answer.")
        notes_text = "\n\n---\n\n".join(state.get("research_notes", []))
        response = llm.invoke([
            SystemMessage(content=SYNTHESIZER_SYSTEM),
            HumanMessage(
                content=(
                    f"Query: {state['query']}\n\n"
                    f"Research notes:\n{notes_text[:4000]}\n\n"
                    "Write the final comprehensive answer."
                )
            ),
        ])
        return {
            **state,
            "final_answer": response.content,
            "messages": state["messages"] + [response],
        }

    def should_continue_research(state: AgentState) -> Literal["tools", "critic"]:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "critic"

    def should_revise(state: AgentState) -> Literal["researcher", "synthesizer"]:
        critique = state.get("critique", "")
        iterations = state.get("iterations", 0)
        if "NEEDS_REVISION" in critique and iterations < MAX_ITERATIONS:
            logger.info("[Router] Routing back to researcher for revision.")
            return "researcher"
        return "synthesizer"

    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("tools", tool_node)
    graph.add_node("extract", extract_tool_results)
    graph.add_node("critic", critic_node)
    graph.add_node("synthesizer", synthesizer_node)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "researcher")
    graph.add_conditional_edges("researcher", should_continue_research)
    graph.add_edge("tools", "extract")
    graph.add_edge("extract", "critic")
    graph.add_conditional_edges("critic", should_revise)
    graph.add_edge("synthesizer", END)

    return graph.compile()