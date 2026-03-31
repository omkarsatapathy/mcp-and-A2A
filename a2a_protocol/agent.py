"""
agent.py — Entry point for the Research Agent.

Exposes a single function `run_research_agent(user_query)` that:
1. Takes a natural language research request
2. Runs the full LangGraph pipeline (parse → search → dedup → summarize)
3. Returns the final research output string

Usage:
    from a2a_protocol.agent import run_research_agent

    result = run_research_agent(
        "I want a research output of 500 words on the environmental impact of coal based powerplants"
    )
    print(result)
"""

import logging
from a2a_protocol.graph import compiled_graph

logger = logging.getLogger(__name__)


def run_research_agent(user_query: str) -> str:
    """
    Execute the full research pipeline for a given user query.

    Args:
        user_query: Natural language research request, e.g.
            "I want a research output of 500 words on the environmental impact
             of coal based powerplants"

    Returns:
        The final research summary as a plain text string.

    Raises:
        ValueError: If the query is empty or parsing fails.
        Exception: If any pipeline stage fails (search, scrape, LLM call, etc.)
    """
    if not user_query or not user_query.strip():
        raise ValueError("user_query cannot be empty")

    logger.info("=" * 80)
    logger.info("RESEARCH AGENT: Pipeline starting")
    logger.info(f"  Query: {user_query[:100]}{'...' if len(user_query) > 100 else ''}")
    logger.info("=" * 80)

    # Run the compiled LangGraph with the user query as initial state
    initial_state = {"user_query": user_query}
    final_state = compiled_graph.invoke(initial_state)

    research_output = final_state.get("research_output", "")

    word_count = len(research_output.split()) if research_output else 0
    logger.info("=" * 80)
    logger.info(f"RESEARCH AGENT: Pipeline complete ({word_count} words)")
    logger.info("=" * 80)

    return research_output


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter your research query: ").strip()

    if not query:
        print("Error: Please provide a research query.")
        sys.exit(1)

    result = run_research_agent(query)
    print("\n" + "=" * 80)
    print("RESEARCH OUTPUT")
    print("=" * 80)
    print(result)
