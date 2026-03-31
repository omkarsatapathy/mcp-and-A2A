# config.py — Configuration for the research agent pipeline

import os

# —— Model Tiers ——————————————————————————————————————————————————————————
OPENAI_MODEL_LOW: str = "gpt-5.4-nano"
OPENAI_MODEL_HIGH: str = "gpt-5.4-mini"

# —— LLM Provider Configuration ——————————————————————————————————————————
LLM_CONFIG = {
    "default": {
        "provider":        "openai",
        "model":           OPENAI_MODEL_LOW,
        "secret_key_name": "OPENAI_API_KEY",
        "temperature":     0.7,
    },
    "research": {
        "provider":        "openai",
        "model":           OPENAI_MODEL_LOW,
        "secret_key_name": "OPENAI_API_KEY",
        "temperature":     0.2,
    },
    "summarizer": {
        "provider":        "openai",
        "model":           OPENAI_MODEL_HIGH,
        "secret_key_name": "OPENAI_API_KEY",
        "temperature":     0.4,
    },
}

# —— Semantic Dedup & Rerank Config ————————————————————————————————————————
EMBEDDING_MODEL         = "all-MiniLM-L6-v2"
RERANKER_MODEL          = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LOCAL_MODEL_DIR         = "/opt/models"
CHUNK_SIZE              = 500       # words
CHUNK_OVERLAP           = 100       # words
MIN_CHUNK_WORDS         = 50
SIMILARITY_THRESHOLD    = 0.85
TOP_K_CHUNKS            = 25        # top-k chunks after reranking against the single topic

# —— Tavily Search Config ——————————————————————————————————————————————————
TAVILY_MAX_RESULTS      = 10
SCRAPE_WORKERS          = 15
MIN_CONTENT_LEN         = 300
