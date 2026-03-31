"""
Summarizer Node — LLM-powered research summary generation.

Takes the ranked chunks from the dedup node and produces a coherent,
well-structured research summary at the user's desired word count.
"""

import logging
from typing import Any, Dict

from a2a_protocol.llm_factory.factory import get_llm
from a2a_protocol.state import ResearchState

logger = logging.getLogger(__name__)


# ── System prompt ───────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a senior research analyst who produces clear, comprehensive research summaries.

You will receive:
1. A RESEARCH TOPIC
2. A TARGET WORD COUNT
3. A RESEARCH CORPUS (ranked chunks of web-scraped content)

YOUR TASK:
Write a well-structured research summary on the given topic using ONLY the facts \
and information present in the provided corpus. Do not fabricate or hallucinate \
any claims.

REQUIREMENTS:
- Length: Match the target word count as closely as possible (within 10% tolerance).
- Structure: Use clear paragraphs. Start with an overview/introduction, cover key \
  findings and perspectives in the body, and end with a brief conclusion or outlook.
- Tone: Objective, analytical, professional. Suitable for an informed reader.
- Cite specific data points, statistics, and expert opinions from the corpus.
- Do not use markdown formatting, bullet points, or headers. Write in flowing \
  prose paragraphs only.
- Do not add meta-commentary like "Based on the research..." or "In conclusion...". \
  Just deliver the content directly.
- If the corpus lacks sufficient information on certain aspects, acknowledge the \
  gap briefly rather than inventing details.
"""


# ── Node entry point ───────────────────────────────────────────────────────────

def run(state: ResearchState) -> Dict[str, Any]:
    """
    Summarizer Node — produce a research summary from ranked chunks.

    Args:
        state: ResearchState with ranked_chunks, topic, and desired_length populated.

    Returns:
        {"research_output": <final summary string>}
    """
    logger.info("=" * 80)
    logger.info("SUMMARIZER NODE: Starting")
    logger.info("=" * 80)

    ranked_chunks = state.ranked_chunks
    topic = state.topic
    desired_length = state.desired_length

    if not ranked_chunks:
        logger.warning("  ranked_chunks is empty — returning empty output")
        return {"research_output": "No research data available to summarize."}

    # Flatten chunks into a single corpus string
    corpus = "\n\n".join(c["text"] for c in ranked_chunks)
    total_chunks = len(ranked_chunks)
    logger.info(
        f"  Input: {total_chunks} chunks, {len(corpus):,} chars total"
    )
    logger.info(f"  Topic: '{topic}', Target length: {desired_length} words")

    # Build user message
    user_prompt = (
        f"RESEARCH TOPIC: {topic}\n"
        f"TARGET WORD COUNT: {desired_length}\n\n"
        f"RESEARCH CORPUS:\n{'=' * 60}\n{corpus}\n{'=' * 60}"
    )

    # Estimate max tokens needed (1 word ~ 1.5 tokens, plus margin).
    # Reasoning models (e.g. gpt-5.4-*) consume internal "thinking" tokens from
    # the same budget, so we need a generous floor to avoid starving the output.
    max_tokens = max(int(desired_length * 2), 1024)

    llm = get_llm("summarizer")
    logger.info(f"  Calling {llm.get_provider_name()} ({llm.model}) for summarization...")

    response = llm.generate(
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=max_tokens,
    )

    research_output = response.content.strip()
    word_count = len(research_output.split())

    logger.info(
        f"  Summary generated: {word_count} words "
        f"(target: {desired_length}, tokens: {response.input_tokens}->{response.output_tokens})"
    )
    logger.info("=" * 80)
    logger.info("SUMMARIZER NODE: Complete")
    logger.info("=" * 80)

    return {"research_output": research_output}
