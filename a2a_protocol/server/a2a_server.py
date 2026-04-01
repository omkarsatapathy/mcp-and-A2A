"""
a2a_server.py — A2A v1.0 server entry point.

Exposes the research agent as a discoverable A2A service:
  - GET  /.well-known/agent-card.json  -> AgentCard (RFC 8615 discovery)
  - POST /                             -> JSON-RPC 2.0 (message/send, message/stream)
  - GET  /ping                         -> Health check

Usage:
    uvicorn a2a_protocol.server.a2a_server:app --host 0.0.0.0 --port 8080
"""

import logging
import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from starlette.requests import Request
from starlette.responses import JSONResponse

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from a2a_protocol.server.agent_executor import ResearchAgentExecutor

HOST = os.getenv("A2A_HOST", "0.0.0.0")
PORT = int(os.getenv("A2A_PORT", "8080"))
BASE_URL = os.getenv("A2A_BASE_URL", f"http://localhost:{PORT}")

# -- Agent Skill --

skill = AgentSkill(
    id="research",
    name="Deep Research",
    description=(
        "Search, scrape, deduplicate, and summarize web content on any topic. "
        "Returns a sourced, coherent research summary."
    ),
    tags=["research", "web", "summarization"],
    examples=[
        "Explain quantum computing in simple terms",
        "What are the latest advances in renewable energy?",
    ],
    input_modes=["text/plain"],
    output_modes=["text/plain"],
)

# -- Agent Card --

agent_card = AgentCard(
    name="Research Agent",
    description="LangGraph research pipeline: search -> scrape -> dedup -> summarize.",
    url=BASE_URL,
    version="1.0.0",
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    capabilities=AgentCapabilities(streaming=False),
    skills=[skill],
)

# -- Request Handler & Server --

request_handler = DefaultRequestHandler(
    agent_executor=ResearchAgentExecutor(),
    task_store=InMemoryTaskStore(),
)

server = A2AStarletteApplication(
    agent_card=agent_card,
    http_handler=request_handler,
)

app = server.build()


# -- Health check --

async def ping(request: Request) -> JSONResponse:
    return JSONResponse({"status": "healthy"})


app.add_route("/ping", ping, methods=["GET"])

# AgentCore routes all POST /runtimes/{arn}/invocations to the container's
# /invocations path. Mirror the A2A JSON-RPC handler there so both local (/)
# and AgentCore (/invocations) work without changing anything else.
app.add_route("/invocations", server._handle_requests, methods=["POST"])


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
