"""
agent_executor.py — A2A v1.0 AgentExecutor bridging the protocol to the LangGraph pipeline.

Receives A2A JSON-RPC requests via RequestContext, runs the research pipeline,
and writes status/artifact events to the EventQueue.

Integrates with Bedrock AgentCore Memory to persist research results and
retrieve prior context for follow-up queries.
"""

import asyncio
import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import (
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from a2a.utils import new_agent_text_message, new_task, new_text_artifact

from a2a_protocol.agent import run_research_agent
from a2a_protocol.services.memory_service import get_memory_store

logger = logging.getLogger(__name__)


class ResearchAgentExecutor(AgentExecutor):
    """Bridges the A2A v1.0 protocol to the LangGraph research pipeline.

    When Bedrock AgentCore Memory is enabled (USE_BEDROCK_MEMORY=true),
    the executor will:
      1. Retrieve prior memories relevant to the query before research.
      2. Store the completed research result as a semantic memory.
    """

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        # 1. Create or retrieve the task
        task = context.current_task or new_task(context.message)
        await event_queue.enqueue_event(task)

        # 2. Signal working state
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                final=False,
                status=TaskStatus(
                    state=TaskState.working,
                    message=new_agent_text_message(
                        "Researching…",
                        context_id=context.context_id,
                        task_id=context.task_id,
                    ),
                ),
            )
        )

        try:
            # 3. Extract user query from the first text Part
            query = context.message.parts[0].root.text
            session_id = context.context_id or context.task_id

            # 4. Retrieve prior memories for context enrichment
            actor_id = "a2a-client"
            prior_context = ""
            try:
                memory_store = get_memory_store()
                loop = asyncio.get_running_loop()
                memories = await loop.run_in_executor(
                    None,
                    memory_store.search_memories,
                    session_id, actor_id, query, 3,
                )
                if memories:
                    snippets = [
                        f"[Prior]: {mem.get('text', '')[:500]}"
                        for mem in memories if mem.get("text")
                    ]
                    if snippets:
                        prior_context = (
                            "\n\n--- PRIOR CONTEXT FROM MEMORY ---\n"
                            + "\n".join(snippets)
                            + "\n--- END PRIOR CONTEXT ---\n\n"
                        )
                        logger.info(
                            "Enriched query with %d prior memories", len(snippets)
                        )
            except Exception:
                logger.exception("Memory retrieval failed")

            # 5. Run the LangGraph pipeline (blocking → offload to thread)
            enriched_query = prior_context + query if prior_context else query
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, run_research_agent, enriched_query
            )

            # 6. Store result in Bedrock AgentCore Memory
            try:
                memory_store = get_memory_store()
                await loop.run_in_executor(
                    None,
                    memory_store.store_research_result,
                    session_id,
                    actor_id,
                    query,
                    result,
                    {"word_count": len(result.split()), "task_id": context.task_id},
                )
            except Exception:
                logger.warning("Failed to store memory — result still returned", exc_info=True)

            # 7. Return the research summary as an Artifact
            await event_queue.enqueue_event(
                TaskArtifactUpdateEvent(
                    task_id=context.task_id,
                    context_id=context.context_id,
                    last_chunk=True,
                    artifact=new_text_artifact(
                        name="research_summary", text=result
                    ),
                )
            )

            # 8. Mark complete
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=context.task_id,
                    context_id=context.context_id,
                    final=True,
                    status=TaskStatus(state=TaskState.completed),
                )
            )

        except Exception as e:
            logger.exception("Research pipeline failed")
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=context.task_id,
                    context_id=context.context_id,
                    final=True,
                    status=TaskStatus(
                        state=TaskState.failed,
                        message=new_agent_text_message(
                            f"Research failed: {e}",
                            context_id=context.context_id,
                            task_id=context.task_id,
                        ),
                    ),
                )
            )

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise Exception("cancel not supported")
