"""
Semantic Dedup & Relevance Ranker Node — Single Topic variant.

Adapted from N03 of the daily_news_automation_via_telegram pipeline.

Three-stage in-memory pipeline:
  Stage 1 — Chunking: Split corpus into overlapping chunks.
  Stage 2 — Semantic Deduplication: FAISS cosine-sim dedup.
  Stage 3 — CrossEncoder Reranking: Score & rank against the single research topic.

Parallel optimisations:
  - warmup(): loads both ML models concurrently; call during the preceding
    research node so models are warm by the time dedup starts.
  - Parallel model loading in run(): reranker loads in a background thread
    while embedding + dedup are executing (improvement #3).

Inference backend: ONNX Runtime (no PyTorch dependency).
  Models are pre-exported to int8 ONNX format via export_models_onnx.py
  and stored at LOCAL_MODEL_DIR/embedding/ and LOCAL_MODEL_DIR/reranker/.

Output state key: ranked_chunks -> List[chunk_dict]
Each chunk_dict has keys: chunk_id, text, word_count, relevance_score
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

import numpy as np
import onnxruntime as ort
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from a2a_protocol.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    LOCAL_MODEL_DIR,
    MIN_CHUNK_WORDS,
    SIMILARITY_THRESHOLD,
    TOP_K_CHUNKS,
)
from a2a_protocol.state import ResearchState

logger = logging.getLogger(__name__)

_EMBEDDING_MODEL_DIR = os.path.join(LOCAL_MODEL_DIR, "embedding")
_RERANKER_MODEL_DIR = os.path.join(LOCAL_MODEL_DIR, "reranker")


# ── ONNX inference wrappers ────────────────────────────────────────────────────

class _OnnxEmbedder:
    """Drop-in for SentenceTransformer backed by ONNX Runtime — no PyTorch."""

    def __init__(self, model_dir: str):
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self._session = ort.InferenceSession(
            os.path.join(model_dir, "model_quantized.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {inp.name for inp in self._session.get_inputs()}

    def encode(self, texts, normalize_embeddings: bool = True,
               show_progress_bar: bool = False, batch_size: int = 64):
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self._tokenizer(
                batch, padding=True, truncation=True,
                max_length=256, return_tensors="np",
            )
            inputs = {k: v for k, v in enc.items() if k in self._input_names}
            token_emb = self._session.run(None, inputs)[0]          # [B, T, D]
            mask = enc["attention_mask"][:, :, np.newaxis].astype(np.float32)
            embeddings = (token_emb * mask).sum(1) / mask.sum(1).clip(1e-9)
            if normalize_embeddings:
                embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True).clip(1e-9)
            all_embeddings.append(embeddings.astype(np.float32))
        return np.vstack(all_embeddings)


class _OnnxReranker:
    """Drop-in for CrossEncoder backed by ONNX Runtime — no PyTorch."""

    def __init__(self, model_dir: str):
        self._tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self._session = ort.InferenceSession(
            os.path.join(model_dir, "model_quantized.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self._input_names = {inp.name for inp in self._session.get_inputs()}

    def predict(self, pairs, show_progress_bar: bool = False, batch_size: int = 32):
        all_scores: list = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i : i + batch_size]
            queries, texts = zip(*batch)
            enc = self._tokenizer(
                list(queries), list(texts),
                padding=True, truncation=True,
                max_length=512, return_tensors="np",
            )
            inputs = {k: v for k, v in enc.items() if k in self._input_names}
            logits = self._session.run(None, inputs)[0]              # [B, 1] or [B, 2]
            scores = logits[:, 0] if logits.shape[-1] == 1 else logits[:, 1]
            all_scores.extend(scores.tolist())
        return np.array(all_scores, dtype=np.float32)


# ── Model cache ────────────────────────────────────────────────────────────────

_embedding_model: Optional[_OnnxEmbedder] = None
_reranker_model: Optional[_OnnxReranker] = None


def _get_model(model_type: str):
    """Return the ONNX embedder or reranker, loading on first use."""
    global _embedding_model, _reranker_model
    if model_type == "embedding":
        if _embedding_model is None:
            logger.info(f"  Loading ONNX embedding model from {_EMBEDDING_MODEL_DIR}")
            _embedding_model = _OnnxEmbedder(_EMBEDDING_MODEL_DIR)
        return _embedding_model
    elif model_type == "reranker":
        if _reranker_model is None:
            logger.info(f"  Loading ONNX reranker model from {_RERANKER_MODEL_DIR}")
            _reranker_model = _OnnxReranker(_RERANKER_MODEL_DIR)
        return _reranker_model
    raise ValueError(f"Unknown model_type: {model_type}")


def warmup() -> None:
    """
    Pre-load both ML models concurrently.

    Call this from the research node (improvement #1) so both models are
    fully loaded by the time the dedup node starts, hiding ~8 s of I/O
    behind research's network wait.
    """
    logger.info("  [dedup warmup] Pre-loading embedding + reranker models in parallel...")
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_embed = pool.submit(_get_model, "embedding")
        f_rerank = pool.submit(_get_model, "reranker")
        f_embed.result()
        f_rerank.result()
    logger.info("  [dedup warmup] Both models ready.")


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
    # Improvement #3: start loading the reranker in the background now —
    # it will be ready (or nearly so) by the time embed + dedup finish.
    import numpy as np
    _reranker_pool = ThreadPoolExecutor(max_workers=1)
    reranker_future = _reranker_pool.submit(_get_model, "reranker")

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

    # Ensure reranker is ready before stage 3 (usually already done by now)
    reranker_future.result()
    _reranker_pool.shutdown(wait=False)

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
