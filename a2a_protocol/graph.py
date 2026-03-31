"""
graph.py — LangGraph Pipeline Assembly for the Research Agent.

Builds a simple 4-node linear DAG:

  input_parser → research → dedup → summarizer → END

Each node reads from and writes to ResearchState.
"""

import logging
from typing import Any, Dict

from langgraph.graph import END, StateGraph

from a2a_protocol.state import ResearchState
from a2a_protocol.nodes import input_parser, research, dedup, summarizer

logger = logging.getLogger(__name__)


def build_graph():
    """
    Wire the 4-node research pipeline and compile the LangGraph.

    Returns a compiled CompiledStateGraph ready to be invoked with a ResearchState.
    """
    g = StateGraph(ResearchState)

    # Register nodes
    g.add_node("input_parser", input_parser.run)
    g.add_node("research", research.run)
    g.add_node("dedup", dedup.run)
    g.add_node("summarizer", summarizer.run)

    # Entry point
    g.set_entry_point("input_parser")

    # Linear edges
    g.add_edge("input_parser", "research")
    g.add_edge("research", "dedup")
    g.add_edge("dedup", "summarizer")
    g.add_edge("summarizer", END)

    return g.compile()


# Compiled graph singleton
compiled_graph = build_graph()
