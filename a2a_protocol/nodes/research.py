"""
Research Node — Web Search & Scrape for a single topic.

Adapted from N02 of the daily_news_automation_via_telegram pipeline.

Two-stage pipeline:
  Stage 1 — Query Producer:
    Invokes the LLM to generate 3-5 targeted search queries for the given topic.
  Stage 2 — Parallel Search + Scrape:
    All queries fired against Tavily Search API simultaneously.
    Every returned URL is scraped in parallel using trafilatura
    (with BeautifulSoup pre-processing to strip noise).
    All extracted page texts are assembled into a single research corpus.
"""

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from a2a_protocol.config import (
    MIN_CONTENT_LEN,
    SCRAPE_WORKERS,
    TAVILY_MAX_RESULTS,
)
from a2a_protocol.llm_factory.factory import get_llm
from a2a_protocol.services.secrets_service import get_secret
from a2a_protocol.state import ResearchState

logger = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}


# ── Pydantic output schema ─────────────────────────────────────────────────────

class SearchQuery(BaseModel):
    query: str = Field(description="The web-search query string")
    rationale: str = Field(description="One-sentence explanation of why this query was chosen")


class QueryPlan(BaseModel):
    queries: List[SearchQuery] = Field(description="3 to 5 search queries for the topic")


# ── System prompt for query generation ──────────────────────────────────────────

_QUERY_SYSTEM_PROMPT = """\
You are a research query strategist. Given a research topic, produce 3 to 5 \
diverse web-search queries that will surface the most comprehensive and \
authoritative information on the topic.

RULES:
- Each query should target a different angle or sub-aspect of the topic.
- Use terms like "latest", "analysis", "impact", "research" to get high-quality results.
- Include the topic keywords in every query but vary the framing.
- Return ONLY a valid JSON object — no markdown fences, no prose, no comments.

JSON schema (strict):
{{
  "queries": [
    {{"query": "...", "rationale": "..."}},
    {{"query": "...", "rationale": "..."}},
    {{"query": "...", "rationale": "..."}}
  ]
}}
"""


# ── Stage 1: Query generation ──────────────────────────────────────────────────

def _generate_queries(topic: str) -> QueryPlan:
    """Call the research LLM to produce targeted search queries for the topic."""
    llm = get_llm("research")

    user_prompt = f"Research topic: {topic}"

    logger.info(f"  Calling {llm.get_provider_name()} ({llm.model}) for query generation...")
    response = llm.generate(
        system_prompt=_QUERY_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        max_tokens=512,
    )

    raw = response.content.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    plan = QueryPlan(**json.loads(raw))
    logger.info(
        f"  Query plan ready ({len(plan.queries)} queries, "
        f"{response.input_tokens}→{response.output_tokens} tokens)"
    )
    return plan


# ── Stage 2a: Tavily search ────────────────────────────────────────────────────

def _tavily_search(query: str, api_key: str, max_results: int = TAVILY_MAX_RESULTS) -> List[str]:
    """Fire one Tavily search query and return the result URLs."""
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "advanced",
        "include_answer": False,
        "include_raw_content": False,
    }
    try:
        resp = httpx.post("https://api.tavily.com/search", json=payload, timeout=30)
        resp.raise_for_status()
        urls = [r["url"] for r in resp.json().get("results", []) if r.get("url")]
        logger.info(f"    [{query[:55]}...] -> {len(urls)} URLs")
        return urls
    except Exception as exc:
        logger.warning(f"    Tavily failed for '{query[:55]}...': {exc}")
        return []


# ── Stage 2b: Page scraping ────────────────────────────────────────────────────

def _preprocess_html(html: str) -> str:
    """Strip navigation, ads, and boilerplate before text extraction."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in ["nav", "footer", "aside", "header", "script", "style", "form"]:
        for el in soup.find_all(tag):
            el.decompose()
    noise = [
        "author", "byline", "metadata", "share", "social", "comment",
        "sidebar", "related", "newsletter", "subscription", "ad",
        "advertisement", "cookie", "consent", "breadcrumb", "pagination",
    ]
    for pattern in noise:
        for el in soup.find_all(class_=re.compile(pattern, re.I)):
            el.decompose()
        for el in soup.find_all(id=re.compile(pattern, re.I)):
            el.decompose()
    return str(soup)


def _scrape_url(url: str) -> str:
    """Scrape one URL and return cleaned article text (or '' on failure)."""
    import trafilatura
    try:
        resp = httpx.get(url, timeout=15, follow_redirects=True, headers=_HEADERS)
        resp.raise_for_status()
        cleaned_html = _preprocess_html(resp.text)
        text = trafilatura.extract(
            cleaned_html,
            include_comments=False,
            include_tables=False,
            include_links=False,
            include_images=False,
            no_fallback=False,
            output_format="txt",
            favor_precision=True,
        )
        if text and len(text) >= MIN_CONTENT_LEN:
            return text.strip()
    except Exception as exc:
        logger.debug(f"    Scrape failed [{url}]: {exc}")
    return ""


def _scrape_all(urls: List[str]) -> List[str]:
    """Scrape all URLs concurrently. Returns only pages with usable content."""
    texts: List[str] = []
    with ThreadPoolExecutor(max_workers=SCRAPE_WORKERS) as executor:
        futures = {executor.submit(_scrape_url, u): u for u in urls}
        for future in as_completed(futures):
            text = future.result()
            if text:
                texts.append(text)
    return texts


# ── Node entry point ───────────────────────────────────────────────────────────

def run(state: ResearchState) -> Dict[str, Any]:
    """
    Research Node — query generation -> parallel search -> parallel scrape.

    Args:
        state: ResearchState with topic populated by input_parser.

    Returns:
        {"raw_research_data": <assembled corpus string>}
    """
    logger.info("=" * 80)
    logger.info("RESEARCH NODE: Starting")
    logger.info("=" * 80)

    topic = state.topic
    if not topic:
        raise ValueError("topic is empty — cannot run research")

    tavily_api_key = get_secret("TAVILY_API_KEY")

    # Stage 1: Generate queries
    logger.info(f"  Stage 1 — Generating search queries for topic: '{topic}'")
    query_plan = _generate_queries(topic)
    for i, q in enumerate(query_plan.queries, 1):
        logger.info(f"  Q{i}: {q.query}")

    # Stage 2a: Parallel Tavily searches
    logger.info(f"  Stage 2a — Tavily search ({len(query_plan.queries)} queries in parallel)")
    all_urls: List[str] = []
    with ThreadPoolExecutor(max_workers=len(query_plan.queries)) as executor:
        futures = {
            executor.submit(_tavily_search, q.query, tavily_api_key): q
            for q in query_plan.queries
        }
        for future in as_completed(futures):
            all_urls.extend(future.result())

    # Deduplicate URLs
    seen: set = set()
    unique_urls = [u for u in all_urls if not (u in seen or seen.add(u))]
    logger.info(
        f"  {len(unique_urls)} unique URLs after deduplication "
        f"({len(all_urls) - len(unique_urls)} duplicates removed)"
    )

    # Stage 2b: Parallel scraping
    logger.info(f"  Stage 2b — Scraping {len(unique_urls)} pages in parallel")
    page_texts = _scrape_all(unique_urls)
    logger.info(f"  {len(page_texts)}/{len(unique_urls)} pages yielded usable content")

    # Assemble corpus
    separator = "\n\n" + "-" * 80 + "\n\n"
    raw_research_data = separator.join(page_texts)

    logger.info(
        f"  Research corpus assembled: "
        f"{len(raw_research_data):,} chars from {len(page_texts)} pages"
    )
    logger.info("=" * 80)
    logger.info("RESEARCH NODE: Complete")
    logger.info("=" * 80)

    return {"raw_research_data": raw_research_data}
