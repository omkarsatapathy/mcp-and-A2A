"""
memory_service.py — Bedrock AgentCore Memory integration.

Uses the official `bedrock-agentcore` SDK (MemorySessionManager) for
persistent agent memory. Falls back to a local in-process dict when
Bedrock is unavailable (local dev / testing without AWS).

Region: ap-south-1 (Mumbai).

SDK reference:
  pip install bedrock-agentcore
  Control plane: boto3.client('bedrock-agentcore-control')
  Data plane:    boto3.client('bedrock-agentcore')
"""

import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)

AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")
MEMORY_ID = os.getenv("BEDROCK_MEMORY_ID", "")
USE_BEDROCK_MEMORY = os.getenv("USE_BEDROCK_MEMORY", "false").lower() in ("1", "true", "yes")


# ── Bedrock AgentCore Memory Store ──────────────────────────────────────────


class BedrockMemoryStore:
    """
    Wraps the bedrock-agentcore SDK MemorySessionManager.

    Flow:
      1. create_memory_session(actor_id, session_id)
      2. add_turns() — write conversation events → short-term memory
      3. search_long_term_memories() — semantic recall
      4. get_last_k_turns() — recent conversation history
    """

    def __init__(self, memory_id: str | None = None, region: str | None = None):
        self.memory_id = memory_id or MEMORY_ID
        self.region = region or AWS_REGION
        self._manager = None
        self._sessions: dict[str, Any] = {}

    @property
    def manager(self):
        """Lazily initialise the MemorySessionManager."""
        if self._manager is None:
            from bedrock_agentcore.memory import MemorySessionManager

            self._manager = MemorySessionManager(
                memory_id=self.memory_id,
                region_name=self.region,
            )
            logger.info(
                "MemorySessionManager initialised (memory_id=%s, region=%s)",
                self.memory_id,
                self.region,
            )
        return self._manager

    def _get_session(self, actor_id: str, session_id: str):
        """Get or create a memory session (cached per session_id)."""
        if session_id not in self._sessions:
            self._sessions[session_id] = self.manager.create_memory_session(
                actor_id=actor_id,
                session_id=session_id,
            )
            logger.info("Created memory session: actor=%s, session=%s", actor_id, session_id)
        return self._sessions[session_id]

    # ── Write ────────────────────────────────────────────────────────────

    def store_conversation_turn(
        self,
        session_id: str,
        actor_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        """Store a user→assistant conversation turn as memory events."""
        from bedrock_agentcore.memory.constants import ConversationalMessage, MessageRole

        session = self._get_session(actor_id, session_id)
        session.add_turns(
            messages=[
                ConversationalMessage(user_message, MessageRole.USER),
                ConversationalMessage(assistant_message, MessageRole.ASSISTANT),
            ]
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
        from bedrock_agentcore.memory.constants import ConversationalMessage, MessageRole

        # Encode the research as a conversation so strategies can extract facts
        assistant_msg = (
            f"Research completed on: {query}\n\n"
            f"Summary ({metadata.get('word_count', 'N/A')} words):\n"
            f"{result[:2000]}"
        )

        session = self._get_session(actor_id, session_id)
        session.add_turns(
            messages=[
                ConversationalMessage(f"Research request: {query}", MessageRole.USER),
                ConversationalMessage(assistant_msg, MessageRole.ASSISTANT),
            ]
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
        session = self._get_session(actor_id, session_id)
        try:
            records = session.search_long_term_memories(
                query=query,
                namespace_prefix="/",
                top_k=top_k,
            )
            logger.info("Retrieved %d memories for query='%s'", len(records), query[:80])
            return [{"text": str(r)} for r in records]
        except Exception:
            logger.warning("search_long_term_memories failed", exc_info=True)
            return []

    def get_recent_turns(
        self,
        session_id: str,
        actor_id: str,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """Get the last k conversation turns for a session."""
        session = self._get_session(actor_id, session_id)
        try:
            turns = session.get_last_k_turns(k=k)
            return [{"text": str(t)} for t in turns]
        except Exception:
            logger.warning("get_last_k_turns failed", exc_info=True)
            return []


# ── Local Fallback (for dev/testing without AWS) ────────────────────────────


class LocalMemoryStore:
    """
    Simple in-process memory store for local development.
    Mimics the BedrockMemoryStore interface with a plain dict.
    No persistence across restarts — purely for testing the integration.
    """

    def __init__(self):
        self._store: dict[str, list[dict[str, Any]]] = {}
        logger.info("LocalMemoryStore initialised (no AWS — dev mode)")

    def store_conversation_turn(
        self,
        session_id: str,
        actor_id: str,
        user_message: str,
        assistant_message: str,
    ) -> None:
        self._store.setdefault(session_id, []).append({
            "type": "turn",
            "actor_id": actor_id,
            "user": user_message,
            "assistant": assistant_message,
            "timestamp": time.time(),
        })

    def store_research_result(
        self,
        session_id: str,
        actor_id: str,
        query: str,
        result: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._store.setdefault(session_id, []).append({
            "type": "research",
            "actor_id": actor_id,
            "query": query,
            "result": result[:2000],
            "metadata": metadata or {},
            "timestamp": time.time(),
        })
        logger.info("LocalMemory: stored research for '%s'", query[:80])

    def search_memories(
        self,
        session_id: str,
        actor_id: str,
        query: str,
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Simple substring match over stored entries."""
        results = []
        query_lower = query.lower()
        for entries in self._store.values():
            for entry in entries:
                text = entry.get("result", "") or entry.get("assistant", "")
                if query_lower in text.lower() or query_lower in entry.get("query", "").lower():
                    results.append({"text": text[:500], "query": entry.get("query", "")})
                    if len(results) >= top_k:
                        return results
        return results

    def get_recent_turns(
        self,
        session_id: str,
        actor_id: str,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        entries = self._store.get(session_id, [])
        return [{"text": e.get("assistant", e.get("result", ""))} for e in entries[-k:]]


# ── Factory ─────────────────────────────────────────────────────────────────

_memory_store = None


def get_memory_store() -> BedrockMemoryStore | LocalMemoryStore:
    """
    Return the singleton memory store.

    Uses BedrockMemoryStore when USE_BEDROCK_MEMORY=true and BEDROCK_MEMORY_ID
    is set, otherwise falls back to LocalMemoryStore for dev/testing.
    """
    global _memory_store
    if _memory_store is None:
        if USE_BEDROCK_MEMORY and MEMORY_ID:
            logger.info("Using Bedrock AgentCore Memory (id=%s, region=%s)", MEMORY_ID, AWS_REGION)
            _memory_store = BedrockMemoryStore()
        else:
            logger.info(
                "Bedrock Memory disabled or BEDROCK_MEMORY_ID not set — using LocalMemoryStore"
            )
            _memory_store = LocalMemoryStore()
    return _memory_store
