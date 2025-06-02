#!/usr/bin/env python3
"""
CLI runner for the LLM Research Assistant.

Usage:
    python scripts/run_research.py "What are the key advances in LLM agents?"
    python scripts/run_research.py "Explain RAG vs fine-tuning" --verbose
"""

import argparse
import sys
import logging
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.graph import build_research_graph
from src.rag.pipeline import RAGPipeline
from src.utils.helpers import setup_logging


def main():
    parser = argparse.ArgumentParser(description="LLM Research Assistant CLI")
    parser.add_argument("query", type=str, help="Research question to answer")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show agent steps")
    args = parser.parse_args()

    setup_logging(logging.DEBUG if args.verbose else logging.WARNING)

    print(f"\n🔬 Research Assistant\n{'─'*50}")
    print(f"Query: {args.query}\n")

    rag = RAGPipeline()
    graph = build_research_graph(rag)

    initial_state = {
        "messages": [],
        "query": args.query,
        "sub_questions": [],
        "research_notes": [],
        "critique": "",
        "final_answer": "",
        "iterations": 0,
    }

    print("⏳ Running agents...\n")
    for step in graph.stream(initial_state):
        node = list(step.keys())[0]
        icons = {"planner": "🧭", "researcher": "🔍", "tools": "🛠️",
                 "critic": "🧐", "synthesizer": "✍️"}
        print(f"  {icons.get(node, '⚙️')}  {node.capitalize()}...")

    # Get final answer from last step
    final = list(step.values())[0]
    answer = final.get("final_answer", "No answer generated.")

    print(f"\n{'─'*50}\n📋 Answer:\n\n{answer}\n")


if __name__ == "__main__":
    main()
