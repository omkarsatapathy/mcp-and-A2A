"""
tools.py — Wraps the LangGraph research pipeline as a LangChain tool.

The outer LangChain agent calls this tool with explicit (query, word_count)
parameters, bypassing the old input_parser LLM call entirely.
"""

import logging

from langchain_core.tools import tool

from a2a_protocol.pipeline.graph import compiled_graph

logger = logging.getLogger(__name__)


@tool
def research_pipeline(query: str, word_count: int = 500) -> str:
    """Search, scrape, deduplicate, and summarize web content on any topic.

    Args:
        query: The research topic to investigate.
        word_count: Desired output length in words (default 500).

    Returns:
        A sourced, coherent research summary.
    """
    logger.info(
        "research_pipeline invoked  query=%r  word_count=%d",
        query[:80], word_count,
    )

    initial_state = {
        "user_query": query,
        "topic": query,
        "desired_length": word_count,
    }

    final_state = compiled_graph.invoke(initial_state)

    result = final_state.get("research_output", "")
    actual_words = len(result.split()) if result else 0
    logger.info("research_pipeline complete  words=%d", actual_words)

    return result or "No research data available to summarize."
