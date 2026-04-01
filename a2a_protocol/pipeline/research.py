"""
Research Node — Web Search & Scrape for a single topic.

Adapted from N02 of the daily_news_automation_via_telegram pipeline.

Two-stage pipeline:
  Stage 1 — Query Producer:
    Invokes the LLM to generate 3-5 targeted search queries for the given topic.
  Stage 2 — Streaming Search + Scrape (producer-consumer):
    Tavily searches run in parallel.  As each search completes it immediately
    feeds its URLs into the scrape pool — scraping starts before all searches
    finish (improvement #2).
  Improvement #1 — Model Preloading:
    Once scraping begins, dedup's ML models are loaded concurrently in the
    background so they are warm by the time the dedup node starts.
"""

import json
import logging
import re
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Set

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field

from a2a_protocol.config import (
    GOOGLE_SEARCH_MAX_RESULTS,
    MIN_CONTENT_LEN,
    SCRAPE_WORKERS,
    TAVILY_MAX_RESULTS,
)
from a2a_protocol.providers.factory import get_llm
from a2a_protocol.services.secrets_service import get_secret
from a2a_protocol.pipeline.state import ResearchState

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


# ── Stage 2a: Google Custom Search (primary) ───────────────────────────────────

def _google_search(
    query: str,
    api_key: str,
    search_engine_id: str,
    max_results: int = GOOGLE_SEARCH_MAX_RESULTS,
) -> List[str]:
    """Fire one Google Custom Search query and return result URLs."""
    try:
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError

        service = build("customsearch", "v1", developerKey=api_key)
        result = service.cse().list(
            q=query,
            cx=search_engine_id,
            num=min(max_results, 10),  # Google CSE hard cap is 10
        ).execute()
        urls = [item["link"] for item in result.get("items", []) if item.get("link")]
        logger.info(f"    [Google] [{query[:55]}...] -> {len(urls)} URLs")
        return urls
    except Exception as exc:
        logger.warning(f"    Google Search failed for '{query[:55]}...': {exc}")
        return []


# ── Stage 2a: Tavily search (fallback) ────────────────────────────────────────

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
        logger.info(f"    [Tavily] [{query[:55]}...] -> {len(urls)} URLs")
        return urls
    except Exception as exc:
        logger.warning(f"    Tavily failed for '{query[:55]}...': {exc}")
        return []


def _search_with_fallback(
    query: str,
    google_api_key: str | None,
    google_cse_id: str | None,
    tavily_api_key: str | None,
) -> List[str]:
    """Try Google Search first; fall back to Tavily if Google fails or is unconfigured."""
    if google_api_key and google_cse_id:
        urls = _google_search(query, google_api_key, google_cse_id)
        if urls:
            return urls
        logger.warning(f"    Google returned no results for '{query[:55]}...', falling back to Tavily")

    if tavily_api_key:
        return _tavily_search(query, tavily_api_key)

    logger.error(f"    No search backend available for '{query[:55]}...'")
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


def _scrape_streaming(
    search_futures: Dict[Future, Any],
    scrape_executor: ThreadPoolExecutor,
) -> tuple[List[str], Set[str]]:
    """
    Producer-consumer scrape: submit scrape jobs as each Tavily search lands
    rather than waiting for all searches to finish first (improvement #2).

    Args:
        search_futures: mapping of {Future -> SearchQuery} from the Tavily pool.
        scrape_executor: shared ThreadPoolExecutor for scraping.

    Returns:
        List of non-empty page texts.
    """
    seen_urls: Set[str] = set()
    scrape_futures: Dict[Future, str] = {}
    texts: List[str] = []

    # As each Tavily search completes, immediately queue its URLs for scraping.
    for search_future in as_completed(search_futures):
        urls = search_future.result()
        for url in urls:
            if url not in seen_urls:
                seen_urls.add(url)
                f = scrape_executor.submit(_scrape_url, url)
                scrape_futures[f] = url

    # Collect scrape results.
    for scrape_future in as_completed(scrape_futures):
        text = scrape_future.result()
        if text:
            texts.append(text)

    return texts, seen_urls


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

    # Resolve search credentials — Google is primary, Tavily is fallback.
    # All keys go through get_secret() so they work with both .env and AWS SM.
    try:
        google_api_key = get_secret("GOOGLE_SEARCH_API_KEY")
    except ValueError:
        google_api_key = None
    try:
        google_cse_id = get_secret("GOOGLE_SEARCH_ENGINE_ID")
    except ValueError:
        google_cse_id = None
    try:
        tavily_api_key = get_secret("TAVILY_API_KEY")
    except ValueError:
        tavily_api_key = None

    if google_api_key and google_cse_id:
        logger.info("  Search backend: Google Custom Search (Tavily as fallback)")
    elif tavily_api_key:
        logger.info("  Search backend: Tavily only (Google not configured)")
    else:
        raise RuntimeError(
            "No search backend configured. "
            "Set GOOGLE_SEARCH_API_KEY + GOOGLE_SEARCH_ENGINE_ID, or TAVILY_API_KEY."
        )

    # Stage 1: Generate queries
    logger.info(f"  Stage 1 — Generating search queries for topic: '{topic}'")
    query_plan = _generate_queries(topic)
    for i, q in enumerate(query_plan.queries, 1):
        logger.info(f"  Q{i}: {q.query}")

    n_queries = len(query_plan.queries)
    logger.info(
        f"  Stage 2 — Streaming search+scrape "
        f"({n_queries} queries, scrape pool={SCRAPE_WORKERS} workers)"
    )

    # Improvement #1: kick off ML model preloading in the background now —
    # both models will be warm by the time the dedup node starts.
    from a2a_protocol.pipeline import dedup as _dedup_node
    warmup_executor = ThreadPoolExecutor(max_workers=1)
    warmup_future = warmup_executor.submit(_dedup_node.warmup)

    # Improvement #2: producer-consumer — scraping starts as each search lands.
    with ThreadPoolExecutor(max_workers=n_queries) as search_pool, \
         ThreadPoolExecutor(max_workers=SCRAPE_WORKERS) as scrape_pool:

        search_futures = {
            search_pool.submit(
                _search_with_fallback, q.query, google_api_key, google_cse_id, tavily_api_key
            ): q
            for q in query_plan.queries
        }
        page_texts, seen_urls = _scrape_streaming(search_futures, scrape_pool)

    unique_count = len(seen_urls)
    logger.info(
        f"  {unique_count} unique URLs discovered; "
        f"{len(page_texts)} pages yielded usable content"
    )

    # Ensure warmup completed (it should be done well before this point).
    warmup_future.result()
    warmup_executor.shutdown(wait=False)

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
