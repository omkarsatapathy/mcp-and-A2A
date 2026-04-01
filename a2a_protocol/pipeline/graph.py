"""
graph.py — LangGraph Pipeline Assembly for the Research Agent.

Builds a 3-node linear DAG:

  research → dedup → summarizer → END

The caller must pre-populate `topic` and `desired_length` in the initial state.
(The old input_parser LLM call is no longer needed — the outer LangChain agent
provides structured parameters directly.)
"""

import logging

from langgraph.graph import END, StateGraph

from a2a_protocol.pipeline.state import ResearchState
from a2a_protocol.pipeline import research, dedup, summarizer

logger = logging.getLogger(__name__)


def build_graph():
    """
    Wire the 3-node research pipeline and compile the LangGraph.

    Expects initial state with `topic` and `desired_length` already set.
    Returns a compiled CompiledStateGraph.
    """
    g = StateGraph(ResearchState)

    g.add_node("research", research.run)
    g.add_node("dedup", dedup.run)
    g.add_node("summarizer", summarizer.run)

    g.set_entry_point("research")

    g.add_edge("research", "dedup")
    g.add_edge("dedup", "summarizer")
    g.add_edge("summarizer", END)

    return g.compile()


# Compiled graph singleton
compiled_graph = build_graph()
