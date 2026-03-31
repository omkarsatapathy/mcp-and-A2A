"""
Semantic Dedup & Relevance Ranker Node — Single Topic variant.

Adapted from N03 of the daily_news_automation_via_telegram pipeline.

Three-stage in-memory pipeline:
  Stage 1 — Chunking: Split corpus into overlapping chunks.
  Stage 2 — Semantic Deduplication: FAISS cosine-sim dedup.
  Stage 3 — CrossEncoder Reranking: Score & rank against the single research topic.

Output state key: ranked_chunks -> List[chunk_dict]
Each chunk_dict has keys: chunk_id, text, word_count, relevance_score
"""

import logging
import os
from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from a2a_protocol.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    LOCAL_MODEL_DIR,
    MIN_CHUNK_WORDS,
    RERANKER_MODEL,
    SIMILARITY_THRESHOLD,
    TOP_K_CHUNKS,
)
from a2a_protocol.state import ResearchState

logger = logging.getLogger(__name__)

# ── Model cache ────────────────────────────────────────────────────────────────
_embedding_local = os.path.join(LOCAL_MODEL_DIR, EMBEDDING_MODEL)
_reranker_local = os.path.join(LOCAL_MODEL_DIR, RERANKER_MODEL)

_embed_source = _embedding_local if os.path.isdir(_embedding_local) else EMBEDDING_MODEL
_rerank_source = _reranker_local if os.path.isdir(_reranker_local) else RERANKER_MODEL

_embedding_model = None
_reranker_model = None


def _get_model(model_type: str):
    """Return the SentenceTransformer or CrossEncoder, loading on first use."""
    from sentence_transformers import CrossEncoder, SentenceTransformer
    global _embedding_model, _reranker_model
    if model_type == "embedding":
        if _embedding_model is None:
            logger.info(f"  Loading embedding model: {_embed_source}")
            _embedding_model = SentenceTransformer(_embed_source)
        return _embedding_model
    elif model_type == "reranker":
        if _reranker_model is None:
            logger.info(f"  Loading reranker model: {_rerank_source}")
            _reranker_model = CrossEncoder(_rerank_source)
        return _reranker_model
    raise ValueError(f"Unknown model_type: {model_type}")


# ── Stage 1: Chunking ─────────────────────────────────────────────────────────

def _chunk_text(text: str) -> List[Dict[str, Any]]:
    """Split corpus into overlapping word-bounded chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE * 5,
        chunk_overlap=CHUNK_OVERLAP * 5,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks: List[Dict[str, Any]] = []
    for i, chunk in enumerate(splitter.split_text(text)):
        word_count = len(chunk.split())
        if word_count >= MIN_CHUNK_WORDS:
            chunks.append(
                {
                    "chunk_id": f"chunk_{i:04d}",
                    "text": chunk,
                    "word_count": word_count,
                }
            )
    return chunks


# ── Stage 2: Semantic deduplication ───────────────────────────────────────────

def _deduplicate(
    chunks: List[Dict[str, Any]], embeddings
) -> tuple[List[Dict[str, Any]], Any]:
    """Drop near-duplicate chunks via FAISS cosine-similarity search."""
    import faiss
    import numpy as np
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings.astype("float32"))

    keep = np.ones(len(chunks), dtype=bool)
    for i in range(len(chunks)):
        if not keep[i]:
            continue
        k = min(20, len(chunks))
        sims, idxs = index.search(embeddings[i : i + 1].astype("float32"), k)
        for sim, idx in zip(sims[0], idxs[0]):
            if idx != i and keep[idx] and sim >= SIMILARITY_THRESHOLD:
                keep[idx] = chunks[idx]["word_count"] > chunks[i]["word_count"]

    unique_chunks = [c for j, c in enumerate(chunks) if keep[j]]
    return unique_chunks, embeddings[keep]


# ── Stage 3: CrossEncoder reranking (single topic) ────────────────────────────

def _rerank_for_topic(chunks: List[Dict[str, Any]], topic: str) -> List[Dict[str, Any]]:
    """Score every chunk against the research topic and return top-k."""
    if not chunks:
        return []

    pairs = [[topic, c["text"]] for c in chunks]
    scores = _get_model("reranker").predict(pairs, show_progress_bar=False)

    ranked = [
        {**chunk, "relevance_score": float(score)}
        for chunk, score in zip(chunks, scores)
    ]
    ranked.sort(key=lambda x: x["relevance_score"], reverse=True)
    return ranked[:TOP_K_CHUNKS]


# ── Node entry point ──────────────────────────────────────────────────────────

def run(state: ResearchState) -> Dict[str, Any]:
    """
    Semantic Dedup & Relevance Ranker for a single topic.

    Reads raw_research_data from state, runs chunk -> embed -> dedup -> rerank,
    and writes ranked_chunks for the summarizer.

    Args:
        state: ResearchState with raw_research_data and topic populated.

    Returns:
        {"ranked_chunks": [top-k chunks with relevance_score]}
    """
    logger.info("=" * 80)
    logger.info("DEDUP & RERANK NODE: Starting")
    logger.info("=" * 80)

    raw_text = state.raw_research_data
    topic = state.topic

    if not raw_text:
        logger.warning("  raw_research_data is empty — skipped")
        return {"ranked_chunks": []}

    logger.info(f"  Input corpus: {len(raw_text):,} chars")
    logger.info(f"  Reranking against topic: '{topic}'")

    # Stage 1: Chunk
    chunks = _chunk_text(raw_text)
    logger.info(
        f"  Stage 1 — Chunking: {len(chunks)} chunks "
        f"(size~{CHUNK_SIZE}w, overlap~{CHUNK_OVERLAP}w, min={MIN_CHUNK_WORDS}w)"
    )

    if not chunks:
        logger.warning("  No usable chunks produced — returning empty")
        return {"ranked_chunks": []}

    # Stage 2: Embed + Deduplicate
    import numpy as np
    texts = [c["text"] for c in chunks]
    embeddings: np.ndarray = _get_model("embedding").encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    )
    logger.info(f"  Stage 2a — Embeddings: {embeddings.shape}")

    unique_chunks, _ = _deduplicate(chunks, embeddings)
    dropped = len(chunks) - len(unique_chunks)
    logger.info(
        f"  Stage 2b — Dedup: {len(chunks)} -> {len(unique_chunks)} unique chunks "
        f"({dropped} dropped, threshold={SIMILARITY_THRESHOLD})"
    )

    # Stage 3: CrossEncoder reranking against the single topic
    top_k = _rerank_for_topic(unique_chunks, topic)
    if top_k:
        logger.info(
            f"  Stage 3 — Rerank: {len(top_k)} chunks kept "
            f"(best score: {top_k[0]['relevance_score']:.3f})"
        )
    else:
        logger.info("  Stage 3 — Rerank: no chunks")

    logger.info(
        f"  Summary: {len(chunks)} raw -> {len(unique_chunks)} unique -> {len(top_k)} final"
    )
    logger.info("=" * 80)
    logger.info("DEDUP & RERANK NODE: Complete")
    logger.info("=" * 80)

    return {"ranked_chunks": top_k}
