"""
Input Parser Node — LLM-powered query decomposition.

Takes a raw user query like:
  "I want a research output of 500 words on the environmental impact of coal based powerplants"

And extracts a structured Pydantic-validated output:
  { "topic": "environmental impact of coal based powerplants", "length": 500 }
"""

import json
import logging
import re
from typing import Any, Dict

from pydantic import BaseModel, Field

from a2a_protocol.providers.factory import get_llm
from a2a_protocol.pipeline.state import ResearchState

logger = logging.getLogger(__name__)


# ── Pydantic output schema ─────────────────────────────────────────────────────

class ParsedQuery(BaseModel):
    topic: str = Field(description="The core research topic extracted from the user query")
    length: int = Field(description="Desired output length in number of words", ge=50, le=5000)


# ── System prompt ───────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a query parser. The user will give you a research request in natural language.
Your job is to extract exactly two pieces of information:

1. **topic** — the core research topic (strip away phrasing like "I want", "give me", etc.)
2. **length** — the desired output word count. If the user doesn't specify, default to 500.

Return ONLY a valid JSON object with these two keys. No markdown fences, no commentary.

Example input: "I want a research output of 500 words on the environmental impact of coal based powerplants"
Example output: {"topic": "environmental impact of coal based powerplants", "length": 500}

Example input: "Tell me about quantum computing breakthroughs in 300 words"
Example output: {"topic": "quantum computing breakthroughs", "length": 300}

Example input: "Research the effects of social media on mental health"
Example output: {"topic": "effects of social media on mental health", "length": 500}
"""


# ── Node entry point ───────────────────────────────────────────────────────────

def run(state: ResearchState) -> Dict[str, Any]:
    """
    Parse the raw user query into a structured {topic, length} pair.

    Uses an LLM call with Pydantic validation to extract the two parameters
    that drive the rest of the pipeline.

    Args:
        state: ResearchState with user_query populated.

    Returns:
        {"topic": str, "desired_length": int}
    """
    logger.info("=" * 80)
    logger.info("INPUT PARSER: Starting")
    logger.info("=" * 80)

    user_query = state.user_query
    if not user_query:
        raise ValueError("user_query is empty — nothing to parse")

    llm = get_llm("research")
    logger.info(f"  Calling {llm.get_provider_name()} ({llm.model}) to parse query...")

    response = llm.generate(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_query,
        max_tokens=256,
    )

    raw = response.content.strip()
    # Strip markdown code fences if the model wraps its output
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    parsed = ParsedQuery(**json.loads(raw))

    logger.info(f"  Parsed → topic: '{parsed.topic}', length: {parsed.length} words")
    logger.info(f"  Tokens: {response.input_tokens}→{response.output_tokens}")
    logger.info("=" * 80)
    logger.info("INPUT PARSER: Complete")
    logger.info("=" * 80)

    return {"topic": parsed.topic, "desired_length": parsed.length}
