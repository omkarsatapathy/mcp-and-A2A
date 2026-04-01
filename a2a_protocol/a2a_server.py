"""
a2a_server.py — A2A v1.0 server entry point.

Exposes the research agent as a discoverable A2A service:
  - GET  /.well-known/agent-card.json  → AgentCard (RFC 8615 discovery)
  - POST /                             → JSON-RPC 2.0 (message/send, message/stream)

Bedrock AgentCore Runtime compatibility:
  - GET  /ping                         → Health check (required by AgentCore)
  - POST /invocations                  → AgentCore invocation bridge

Usage:
    python -m a2a_protocol.a2a_server
"""

import logging
import os
import uuid
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from starlette.requests import Request
from starlette.responses import JSONResponse

# Load .env from the same directory as this script (matches run.py behaviour)
load_dotenv(Path(__file__).parent / ".env")

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

from a2a_protocol.agent_executor import ResearchAgentExecutor

load_dotenv()

HOST = os.getenv("A2A_HOST", "0.0.0.0")
PORT = int(os.getenv("A2A_PORT", "8080"))
BASE_URL = os.getenv("A2A_BASE_URL", f"http://localhost:{PORT}")

# ── Agent Skill ──────────────────────────────────────────────────────────────

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

# ── Agent Card ───────────────────────────────────────────────────────────────

agent_card = AgentCard(
    name="Research Agent",
    description="LangGraph research pipeline: search → scrape → dedup → summarize.",
    url=BASE_URL,
    version="1.0.0",
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    capabilities=AgentCapabilities(streaming=True),
    skills=[skill],
)

# ── Request Handler & Server ─────────────────────────────────────────────────

request_handler = DefaultRequestHandler(
    agent_executor=ResearchAgentExecutor(),
    task_store=InMemoryTaskStore(),
)

server = A2AStarletteApplication(
    agent_card=agent_card,
    http_handler=request_handler,
)

app = server.build()


# ── Bedrock AgentCore Runtime Endpoints ─────────────────────────────────────
# AgentCore requires /ping (GET) for health and /invocations (POST) for calls.
# /invocations bridges AgentCore payloads into A2A JSON-RPC message/send.


async def ping(request: Request) -> JSONResponse:
    return JSONResponse({"status": "healthy"})


async def invocations(request: Request) -> JSONResponse:
    """Bridge AgentCore invoke_agent_runtime → A2A message/send.

    Expected payload from AgentCore:
        {"input": {"prompt": "..."}}

    This wraps it into a JSON-RPC message/send call to the A2A handler
    and returns the result.
    """
    try:
        body = await request.json()
        prompt = body.get("input", {}).get("prompt", "")
        if not prompt:
            return JSONResponse(
                {"error": "No prompt found in input. Provide {'input': {'prompt': '...'}}"},
                status_code=400,
            )

        # Build an A2A JSON-RPC message/send request
        a2a_payload = {
            "jsonrpc": "2.0",
            "id": str(uuid.uuid4()),
            "method": "message/send",
            "params": {
                "message": {
                    "messageId": str(uuid.uuid4()),
                    "role": "user",
                    "parts": [{"kind": "text", "text": prompt}],
                }
            },
        }

        # Forward to the A2A handler internally via httpx
        import httpx
        import logging as log

        logger = log.getLogger(__name__)
        logger.info(f"[/invocations] Received prompt: {prompt[:100]}")

        async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{PORT}") as client:
            resp = await client.post(
                "/",
                json=a2a_payload,
                headers={"Content-Type": "application/json"},
                timeout=300.0,
            )

        logger.info(f"[/invocations] A2A response status: {resp.status_code}")
        result = resp.json()

        # Extract the research summary from the A2A response
        task = result.get("result", {})
        artifacts = task.get("artifacts", [])
        output_text = ""
        if artifacts:
            for artifact in artifacts:
                for part in artifact.get("parts", []):
                    if part.get("kind") == "text":
                        output_text += part["text"]

        return JSONResponse({
            "output": {
                "message": output_text or "No result produced",
                "task_id": task.get("id", ""),
                "status": task.get("status", {}).get("state", "unknown"),
            }
        })
    except Exception as e:
        import logging as log
        logger = log.getLogger(__name__)
        logger.exception(f"[/invocations] Error: {e}")
        return JSONResponse(
            {"error": f"Internal error: {str(e)}"},
            status_code=500,
        )


# Register AgentCore routes
app.add_route("/ping", ping, methods=["GET"])
app.add_route("/invocations", invocations, methods=["POST"])


if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)
