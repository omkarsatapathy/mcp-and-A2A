"""
memory_service.py — Bedrock AgentCore Memory integration.

Uses the official `bedrock-agentcore` SDK (MemoryClient) for
persistent agent memory. Set BEDROCK_MEMORY_ID to enable.

Region: ap-south-1 (Mumbai).

SDK reference:
  pip install bedrock-agentcore
  from bedrock_agentcore.memory import MemoryClient
"""

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
MEMORY_ID = os.getenv("BEDROCK_MEMORY_ID", "")


# ── Bedrock AgentCore Memory Store ──────────────────────────────────────────


class BedrockMemoryStore:
    """
    Wraps the bedrock-agentcore SDK MemoryClient.

    Flow:
      1. save_turn(memory_id, actor_id, session_id, user_input, agent_response)
      2. retrieve_memories(memory_id, namespace, query) — semantic recall
      3. get_last_k_turns(memory_id, actor_id, session_id, k) — recent history
    """

    def __init__(self, memory_id: str | None = None, region: str | None = None):
        self.memory_id = memory_id or MEMORY_ID
        self.region = region or AWS_REGION
        self._client = None

    @property
    def client(self):
        """Lazily initialise the MemoryClient."""
        if self._client is None:
            from bedrock_agentcore.memory import MemoryClient

            self._client = MemoryClient(region_name=self.region)
            logger.info(
                "MemoryClient initialised (memory_id=%s, region=%s)",
                self.memory_id,
                self.region,
            )
        return self._client

    # ── Write ────────────────────────────────────────────────────────────

    def store_conversation_turn(
        self,
        session_id: str,
        actor_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """Store a user→assistant conversation turn."""
        self.client.save_turn(
            memory_id=self.memory_id,
            actor_id=actor_id,
            session_id=session_id,
            user_input=user_message,
            agent_response=assistant_message,
        )
        logger.info("Stored conversation turn: session=%s", session_id)

    def store_research_result(
        self,
        session_id: str,
        actor_id: str,
        query: str,
        result: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a research query + result as a conversation turn in memory."""
        word_count = metadata.get("word_count", "N/A") if metadata else "N/A"
        assistant_msg = (
            f"Research completed on: {query}\n\n"
            f"Summary ({word_count} words):\n"
            f"{result[:2000]}"
        )
        self.client.save_turn(
            memory_id=self.memory_id,
            actor_id=actor_id,
            session_id=session_id,
            user_input=f"Research request: {query}",
            agent_response=assistant_msg,
        )
        logger.info("Stored research result in memory: session=%s, topic='%s'", session_id, query[:80])

    # ── Read ─────────────────────────────────────────────────────────────

    def search_memories(
        self,
        session_id: str,
        actor_id: str,
        query: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Semantic search over long-term memories."""
        try:
            records = self.client.retrieve_memories(
                memory_id=self.memory_id,
                namespace="/",
                query=query,
                actor_id=actor_id,
                top_k=top_k,
            )
            logger.info("Retrieved %d memories for query='%s'", len(records), query[:80])
            return [{"text": str(r)} for r in records]
        except Exception:
            logger.warning("retrieve_memories failed", exc_info=True)
            return []

    def get_recent_turns(
        self,
        session_id: str,
        actor_id: str,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """Get the last k conversation turns for a session."""
        try:
            turns = self.client.get_last_k_turns(
                memory_id=self.memory_id,
                actor_id=actor_id,
                session_id=session_id,
                k=k,
            )
            return [{"text": str(t)} for t in turns]
        except Exception:
            logger.warning("get_last_k_turns failed", exc_info=True)
            return []


# ── No-op fallback ──────────────────────────────────────────────────────────

class NoOpMemoryStore:
    """Fallback memory store when BEDROCK_MEMORY_ID is not set."""

    def search_memories(self, session_id: str, actor_id: str, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        return []

    def store_research_result(self, session_id: str, actor_id: str, query: str, result: str, metadata: dict[str, Any] | None = None) -> None:
        pass

    def store_conversation_turn(self, session_id: str, actor_id: str, user_message: str, assistant_message: str) -> None:
        pass

    def get_recent_turns(self, session_id: str, actor_id: str, k: int = 5) -> list[dict[str, Any]]:
        return []


# ── Factory ─────────────────────────────────────────────────────────────────

_memory_store: BedrockMemoryStore | NoOpMemoryStore | None = None


def get_memory_store() -> BedrockMemoryStore | NoOpMemoryStore:
    """
    Return the singleton memory store.

    Returns BedrockMemoryStore if BEDROCK_MEMORY_ID is set, otherwise returns
    a no-op fallback that disables memory features.
    """
    global _memory_store
    if _memory_store is None:
        if not MEMORY_ID:
            logger.warning("BEDROCK_MEMORY_ID not set — memory features disabled")
            _memory_store = NoOpMemoryStore()
        else:
            logger.info("Using Bedrock AgentCore Memory (id=%s, region=%s)", MEMORY_ID, AWS_REGION)
            _memory_store = BedrockMemoryStore()
    return _memory_store
