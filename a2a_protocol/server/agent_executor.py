"""
agent_executor.py — A2A v1.0 AgentExecutor bridging the protocol to the
LangChain research agent.

Receives A2A JSON-RPC requests via RequestContext, runs the LangChain
ReAct agent (which internally calls the LangGraph research pipeline),
and writes status/artifact events to the EventQueue.
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

from a2a_protocol.agent.langchain_agent import run_agent

logger = logging.getLogger(__name__)


class ResearchAgentExecutor(AgentExecutor):
    """Bridges the A2A v1.0 protocol to the LangChain research agent."""

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
                        "Researching...",
                        context_id=context.context_id,
                        task_id=context.task_id,
                    ),
                ),
            )
        )

        try:
            # 3. Extract user query from the first text Part
            query = context.message.parts[0].root.text

            # 4. Run the LangChain agent (blocking -> offload to thread)
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, run_agent, query)

            # 5. Return the research summary as an Artifact
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

            # 6. Mark complete
            await event_queue.enqueue_event(
                TaskStatusUpdateEvent(
                    task_id=context.task_id,
                    context_id=context.context_id,
                    final=True,
                    status=TaskStatus(state=TaskState.completed),
                )
            )

        except Exception as e:
            logger.exception("Research agent failed")
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
