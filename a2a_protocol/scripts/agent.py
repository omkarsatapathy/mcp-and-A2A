"""
agent.py — Entry point for the Research Agent.

Exposes a single function `run_research_agent(user_query)` that:
1. Takes a natural language research request
2. Runs the LangChain ReAct agent (which calls the LangGraph pipeline tool)
3. Returns the final research output string

Usage:
    from a2a_protocol.scripts.agent import run_research_agent

    result = run_research_agent(
        "I want a research output of 500 words on the environmental impact of coal based powerplants"
    )
    print(result)
"""

import logging

from a2a_protocol.agent.langchain_agent import run_agent

logger = logging.getLogger(__name__)


def run_research_agent(user_query: str) -> str:
    """
    Execute the full research pipeline for a given user query.

    Args:
        user_query: Natural language research request.

    Returns:
        The final research summary as a plain text string.
    """
    if not user_query or not user_query.strip():
        raise ValueError("user_query cannot be empty")

    logger.info("=" * 80)
    logger.info("RESEARCH AGENT: Starting")
    logger.info("  Query: %s", user_query[:100])
    logger.info("=" * 80)

    result = run_agent(user_query)

    word_count = len(result.split()) if result else 0
    logger.info("=" * 80)
    logger.info("RESEARCH AGENT: Complete (%d words)", word_count)
    logger.info("=" * 80)

    return result


# -- CLI entry point --

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
