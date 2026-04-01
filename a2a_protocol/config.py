# config.py — Configuration for the research agent pipeline

import os

# —— AWS Region ———————————————————————————————————————————————————————————
AWS_REGION: str = os.getenv("AWS_REGION", "ap-south-1")  # Mumbai

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

# —— Bedrock AgentCore ————————————————————————————————————————————————————
BEDROCK_MEMORY_ID       = os.getenv("BEDROCK_MEMORY_ID", "")          # required
AGENTCORE_NAMESPACE     = os.getenv("AGENTCORE_NAMESPACE", "research-agent")
AGENTCORE_RUNTIME_NAME  = os.getenv("AGENTCORE_RUNTIME_NAME", "ResearchAgentA2A")

# —— Semantic Dedup & Rerank Config ————————————————————————————————————————
EMBEDDING_MODEL         = "all-MiniLM-L6-v2"
RERANKER_MODEL          = "cross-encoder/ms-marco-MiniLM-L-6-v2"
LOCAL_MODEL_DIR         = os.path.join(os.path.dirname(__file__), "models")
CHUNK_SIZE              = 500       # words
CHUNK_OVERLAP           = 100       # words
MIN_CHUNK_WORDS         = 50
SIMILARITY_THRESHOLD    = 0.85
TOP_K_CHUNKS            = 25        # top-k chunks after reranking against the single topic

# —— Google Custom Search Config ——————————————————————————————————————————
GOOGLE_SEARCH_MAX_RESULTS = 10   # max per query (Google CSE cap is 10)

# —— Tavily Search Config (fallback) ——————————————————————————————————————
TAVILY_MAX_RESULTS      = 10
SCRAPE_WORKERS          = 15
MIN_CONTENT_LEN         = 300
