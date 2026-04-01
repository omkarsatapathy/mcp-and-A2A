"""
langchain_agent.py — LangChain agent wrapping the research pipeline tool.

Uses LangChain v1.2's `create_agent` which returns a compiled LangGraph
with tool-calling support. The agent receives a query, calls the
research_pipeline tool, and returns the result.
"""

import logging

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from a2a_protocol.config import OPENAI_MODEL_HIGH
from a2a_protocol.services.secrets_service import get_secret
from a2a_protocol.agent.tools import research_pipeline

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a research agent. When the user asks a research question, "
    "use the research_pipeline tool to produce a comprehensive summary. "
    "Pass the user's topic as the query and extract or default the word count. "
    "Return the tool's output directly as your final answer — do not add "
    "commentary or re-summarize."
)


def _build_agent():
    """Build the LangChain agent with the research pipeline tool."""
    api_key = get_secret("OPENAI_API_KEY")

    llm = ChatOpenAI(
        model=OPENAI_MODEL_HIGH,
        api_key=api_key,
        temperature=0.2,
    )

    agent = create_agent(
        model=llm,
        tools=[research_pipeline],
        system_prompt=_SYSTEM_PROMPT,
    )

    return agent


# Lazy singleton
_agent = None


def get_agent():
    """Return the singleton agent (built on first call)."""
    global _agent
    if _agent is None:
        logger.info("Building LangChain research agent...")
        _agent = _build_agent()
        logger.info("LangChain research agent ready.")
    return _agent


def run_agent(query: str) -> str:
    """Run the research agent synchronously and return the output string."""
    agent = get_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": query}]})

    # Extract the final text from the agent's response messages
    messages = result.get("messages", [])
    if messages:
        last_msg = messages[-1]
        # LangChain message objects have a .content attribute
        content = getattr(last_msg, "content", None) or str(last_msg)
        return content

    return ""
