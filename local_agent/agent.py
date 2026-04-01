#!/usr/bin/env python3
"""
local_agent/agent.py — Google ADK agent (GPT-4o-mini) with proper A2A integration.

A2A best-practice flow (per https://a2a-protocol.org/latest/tutorials/python/)
──────────────────────────────────────────────────────────────────────────────
  1. AgentCard        → constructed from known ARN (card GET not supported via AgentCore proxy)
  2. A2AClient        → sends typed SendMessageRequest (MessageSendParams)
  3. AWSSigV4Auth     → signs every httpx request transparently (botocore)
  4. Task response    → artifacts[*].parts[*].text extracted via SDK types

Run:
    python local_agent/agent.py "What time is it in IST?"
    python local_agent/agent.py "Research quantum computing"
    python local_agent/agent.py          # interactive prompt
"""

import asyncio
import logging
import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import boto3
import botocore.auth
import botocore.awsrequest
import botocore.credentials
import httpx
from dotenv import load_dotenv
from httpx import Auth, Request

load_dotenv(Path(__file__).parent / ".env")

# Official A2A SDK — pip install a2a-sdk
from a2a.client import A2AClient
from a2a.types import AgentCapabilities, AgentCard, MessageSendParams, SendMessageRequest

# Google ADK
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai import types


# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("local_agent")

# Suppress noisy third-party loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)
logging.getLogger("boto3").setLevel(logging.WARNING)
logging.getLogger("google.adk").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


# ── AWS / AgentCore config (from .env) ────────────────────────────────────────

AWS_PROFILE  = os.getenv("AWS_PROFILE", "personal-dev")
AWS_REGION   = os.getenv("AWS_REGION", "ap-south-1")
RUNTIME_NAME = os.getenv("AGENTCORE_RUNTIME_NAME", "ResearchAgentA2A")
RUNTIME_ARN  = os.getenv("AGENTCORE_RUNTIME_ARN", "")   # set to skip ARN discovery
LOCAL_A2A_URL = os.getenv("LOCAL_A2A_URL", "")           # set to test against local server


# ── AWS SigV4 auth for httpx ───────────────────────────────────────────────────

class AWSSigV4Auth(Auth):
    """
    httpx Auth plugin that signs every request with AWS Signature Version 4.

    Pulls credentials from the boto3 session so it respects profiles,
    instance roles, and environment variables — exactly like boto3 itself.
    """

    def __init__(self, session: boto3.Session, service: str = "bedrock-agentcore"):
        frozen = session.get_credentials().get_frozen_credentials()
        self._creds = botocore.credentials.Credentials(
            access_key=frozen.access_key,
            secret_key=frozen.secret_key,
            token=frozen.token,
        )
        self._region  = session.region_name
        self._service = service
        log.debug("AWSSigV4Auth initialised  service=%s  region=%s", service, self._region)

    def auth_flow(self, request: Request):
        # Sign only the minimal stable headers (host + content-type).
        # Signing ALL httpx headers (accept-encoding, connection, content-length…)
        # causes 403 because httpx may independently modify those headers after
        # we sign them, making the canonical request mismatch at AWS.
        minimal_headers = {
            "host": request.url.host,
            "content-type": request.headers.get("content-type", "application/json"),
        }
        aws_req = botocore.awsrequest.AWSRequest(
            method=request.method,
            url=str(request.url),
            data=request.content,
            headers=minimal_headers,
        )
        botocore.auth.SigV4Auth(self._creds, self._service, self._region).add_auth(aws_req)
        # Inject only the auth headers — leave all other httpx headers untouched.
        for key in ("Authorization", "X-Amz-Date", "X-Amz-Security-Token", "X-Amz-Content-SHA256"):
            val = aws_req.headers.get(key)
            if val:
                request.headers[key] = val
        log.debug("SigV4 signed  method=%s  url=%s", request.method, request.url)
        yield request


# ── Helpers ────────────────────────────────────────────────────────────────────

def _resolve_runtime_arn() -> str:
    """Return the AgentCore runtime ARN from env or via list_agent_runtimes."""
    if RUNTIME_ARN:
        log.info("Using runtime ARN from env")
        return RUNTIME_ARN
    log.info("Resolving runtime ARN for '%s' in %s ...", RUNTIME_NAME, AWS_REGION)
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    control = session.client("bedrock-agentcore-control")
    runtimes = control.list_agent_runtimes().get("agentRuntimes", [])
    match = next((r for r in runtimes if r["agentRuntimeName"] == RUNTIME_NAME), None)
    if not match:
        raise RuntimeError(f"AgentCore runtime '{RUNTIME_NAME}' not found in {AWS_REGION}")
    arn = match["agentRuntimeArn"]
    log.info("Resolved ARN: %s", arn)
    return arn


def _agentcore_base_url(arn: str) -> str:
    """
    Build the Bedrock AgentCore A2A base URL.

    The trailing slash is critical:
      .../invocations   → routes to container's POST /invocations  (AgentCore bridge,
                           expects {"input": {"prompt": "..."}} — NOT A2A JSON-RPC)
      .../invocations/  → routes to container's POST /  (real A2A JSON-RPC endpoint)

    ARN must be URL-percent-encoded in the path.
    """
    url = (
        f"https://bedrock-agentcore.{AWS_REGION}.amazonaws.com"
        f"/runtimes/{quote(arn, safe='')}/invocations"
    )
    log.debug("AgentCore A2A base URL: %s", url)
    return url


# ── Tools ──────────────────────────────────────────────────────────────────────

def fetch_time_ist() -> dict:
    """Return the current system date and time converted to IST (UTC+5:30)."""
    log.info("[tool] fetch_time_ist called")
    ist = timezone(timedelta(hours=5, minutes=30))
    now = datetime.now(ist)
    result = {
        "datetime_ist": now.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": "IST (UTC+5:30)",
    }
    log.info("[tool] fetch_time_ist → %s", result["datetime_ist"])
    return result


async def ask_remote_agent(query: str) -> dict:
    """
    Send *query* to the remote A2A research agent on AWS Bedrock AgentCore.

    NOTE: This is an async tool — Google ADK calls it natively inside its
    event loop, so no asyncio.run() wrapper is needed or allowed.

    A2A flow:
      1. Resolve runtime ARN (env or list_agent_runtimes)
      2. Discover agent via AgentCard (/.well-known/agent-card.json)
      3. Send typed SendMessageRequest via A2AClient (MessageSendParams)
      4. Extract text from Task artifacts (artifacts[*].parts[*].text)

    Args:
        query: The research question to send to the remote agent.
    """
    log.info("[tool] ask_remote_agent called  query=%r", query[:80])

    # Local testing mode — talk to a local A2A server directly (no SigV4)
    if LOCAL_A2A_URL:
        base_url = LOCAL_A2A_URL.rstrip("/") + "/"
        log.info("[a2a] Using local A2A URL: %s", base_url)

        agent_card = AgentCard(
            name="LocalResearchAgent",
            description="Local A2A research agent.",
            url=base_url,
            version="1.0.0",
            default_input_modes=["text/plain"],
            default_output_modes=["text/plain"],
            capabilities=AgentCapabilities(streaming=False),
            skills=[],
        )

        async with httpx.AsyncClient(timeout=300.0) as http:
            client = A2AClient(httpx_client=http, agent_card=agent_card)
            msg_id = uuid.uuid4().hex
            req_id = str(uuid.uuid4())
            log.info("[a2a] Sending message/send  req_id=%s  msg_id=%s", req_id, msg_id)
            request = SendMessageRequest(
                id=req_id,
                params=MessageSendParams(**{
                    "message": {
                        "role": "user",
                        "parts": [{"kind": "text", "text": query}],
                        "message_id": msg_id,
                    }
                }),
            )
            response = await client.send_message(request)
            task = response.root.result
            log.info(
                "[a2a] Task received  task_id=%s  state=%s",
                getattr(task, "id", "N/A"),
                getattr(getattr(task, "status", None), "state", "N/A"),
            )
            texts = []
            for artifact in getattr(task, "artifacts", []) or []:
                for part in getattr(artifact, "parts", []) or []:
                    node = getattr(part, "root", part)
                    if hasattr(node, "text"):
                        texts.append(node.text)
            response_text = "\n".join(texts) if texts else str(task)
            log.info("[a2a] Extracted %d text parts  total_chars=%d", len(texts), len(response_text))
            return {"response": response_text}

    # Production mode — talk to AgentCore with SigV4 signing
    boto_session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    base_url = _agentcore_base_url(_resolve_runtime_arn())

    # Step 1 — construct AgentCard directly from the known ARN/URL.
    #
    # Why not use A2ACardResolver?
    # Bedrock AgentCore's /invocations endpoint is POST-only. The resolver sends
    # a GET to /invocations/.well-known/agent-card.json, which returns 400.
    # AgentCard discovery via GET is only available when the agent server is
    # exposed directly (e.g. localhost). When accessing via AgentCore we already
    # know the ARN, so we construct the card ourselves — this is the appropriate
    # pattern for known-identity agents.
    agent_card = AgentCard(
        name=RUNTIME_NAME,
        description="Remote A2A research agent running on AWS Bedrock AgentCore.",
        url=base_url,
        version="1.0.0",
        default_input_modes=["text/plain"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[],
    )
    log.info("[a2a] AgentCard constructed  name=%r  url=%s", agent_card.name, base_url)

    async with httpx.AsyncClient(
        auth=AWSSigV4Auth(boto_session),
        timeout=300.0,
    ) as http:

        # Step 2 — initialise the typed A2A client
        client = A2AClient(httpx_client=http, agent_card=agent_card)

        # Step 3 — build a typed SendMessageRequest (not a raw dict)
        msg_id = uuid.uuid4().hex
        req_id = str(uuid.uuid4())
        log.info("[a2a] Sending message/send  req_id=%s  msg_id=%s", req_id, msg_id)
        request = SendMessageRequest(
            id=req_id,
            params=MessageSendParams(**{
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": query}],
                    "message_id": msg_id,
                }
            }),
        )

        # Step 4 — send and parse the A2A Task response
        response = await client.send_message(request)
        task = response.root.result
        log.info(
            "[a2a] Task received  task_id=%s  state=%s",
            getattr(task, "id", "N/A"),
            getattr(getattr(task, "status", None), "state", "N/A"),
        )

        texts = []
        for artifact in getattr(task, "artifacts", []) or []:
            for part in getattr(artifact, "parts", []) or []:
                node = getattr(part, "root", part)  # unwrap Pydantic discriminated union
                if hasattr(node, "text"):
                    texts.append(node.text)

        response_text = "\n".join(texts) if texts else str(task)
        log.info("[a2a] Extracted %d text parts  total_chars=%d", len(texts), len(response_text))
        return {"response": response_text}


# ── Agent definition ───────────────────────────────────────────────────────────

root_agent = Agent(
    name="local_agent",
    model=LiteLlm(model="openai/gpt-4o-mini"),
    description="Local assistant with IST time lookup and A2A remote-agent access.",
    instruction=(
        "You are a helpful assistant. "
        "Use fetch_time_ist to answer questions about the current date or time. "
        "Use ask_remote_agent to forward research queries to the remote A2A agent."
    ),
    tools=[fetch_time_ist, ask_remote_agent],
)


# ── CLI runner ─────────────────────────────────────────────────────────────────

async def _run(query: str) -> None:
    log.info("=== Starting local_agent  query=%r", query[:80])
    runner = InMemoryRunner(agent=root_agent, app_name="local_agent")
    session = await runner.session_service.create_session(
        app_name="local_agent", user_id="cli"
    )
    log.info("Session created  session_id=%s", session.id)

    content = types.Content(role="user", parts=[types.Part(text=query)])

    async for event in runner.run_async(
        user_id="cli", session_id=session.id, new_message=content
    ):
        if event.is_final_response() and event.content:
            log.info("Final response received")
            print()
            for part in event.content.parts:
                if part.text:
                    print(part.text)


if __name__ == "__main__":
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Query: ").strip()
    if not query:
        print("No query provided.", file=sys.stderr)
        sys.exit(1)
    asyncio.run(_run(query))
