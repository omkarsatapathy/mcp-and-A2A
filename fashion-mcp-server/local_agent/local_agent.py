"""
Fashion Agent — local runner
  LLM    : OpenAI (via LiteLLM inside Google ADK)
  Tools  : fashion-mcp-server on AWS AgentCore Runtime
  Auth   : AWS SigV4 via botocore (uses your local ~/.aws credentials)
  Transport: Direct HTTPS to AgentCore with Accept: application/json, text/event-stream
"""

import asyncio
import inspect
import json
import os
import urllib.parse
import uuid

import boto3
import requests
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from dotenv import load_dotenv
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
AGENTCORE_ARN  = os.environ["AGENTCORE_ARN"]
AWS_REGION     = "ap-southeast-2"
DP_ENDPOINT    = f"https://bedrock-agentcore.{AWS_REGION}.amazonaws.com"

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY  # picked up by LiteLLM

# SigV4 credentials — reads from ~/.aws/credentials or env vars automatically
_session     = boto3.Session()
_credentials = _session.get_credentials()


# ── MCP transport: direct HTTPS + SigV4 + correct Accept header ─────────────

def _mcp_request(method: str, params: dict | None = None, session_id: str | None = None) -> dict:
    """
    Send one MCP JSON-RPC 2.0 message to the AgentCore Runtime endpoint.
    MCP streamable-http requires Accept: application/json, text/event-stream.
    """
    session_id = session_id or str(uuid.uuid4())
    payload    = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}})
    url        = f"{DP_ENDPOINT}/runtimes/{urllib.parse.quote(AGENTCORE_ARN, safe='')}/invocations"

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream",
        "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": session_id,
    }

    # SigV4 sign
    creds       = _credentials.get_frozen_credentials()
    aws_req     = AWSRequest(method="POST", url=url + "?qualifier=DEFAULT", data=payload, headers=headers)
    SigV4Auth(creds, "bedrock-agentcore", AWS_REGION).add_auth(aws_req)
    signed_hdrs = dict(aws_req.headers)

    resp = requests.post(
        url,
        params={"qualifier": "DEFAULT"},
        headers=signed_hdrs,
        data=payload,
        timeout=60,
        stream=True,
    )
    resp.raise_for_status()

    # Response is SSE (text/event-stream) — extract data: line
    content_type = resp.headers.get("content-type", "")
    if "text/event-stream" in content_type:
        for line in resp.iter_lines():
            if line and line.startswith(b"data:"):
                return json.loads(line[5:].strip())
        return {}
    else:
        return resp.json()


def _call_tool_sync(tool_name: str, arguments: dict) -> str:
    """Call one MCP tool synchronously and return text result."""
    import time
    t0 = time.perf_counter()
    result = _mcp_request("tools/call", {"name": tool_name, "arguments": arguments})
    elapsed_ms = (time.perf_counter() - t0) * 1000
    print(f"[MCP] {tool_name} — {elapsed_ms:.0f} ms", flush=True)
    content = result.get("result", {}).get("content", [])
    texts   = [c["text"] for c in content if c.get("type") == "text" and "text" in c]
    return "\n".join(texts) if texts else json.dumps(result.get("result", result), indent=2)


# ── ADK tool factory ─────────────────────────────────────────────────────────

_TYPE_MAP = {"string": str, "number": float, "integer": int, "boolean": bool}

def make_adk_tool(mcp_tool: dict) -> FunctionTool:
    name        = mcp_tool["name"]
    description = mcp_tool.get("description", name).strip()
    schema      = mcp_tool.get("inputSchema", {})
    props       = schema.get("properties", {})
    required    = set(schema.get("required", []))

    params = []
    for param_name, prop in props.items():
        annotation = _TYPE_MAP.get(prop.get("type", "string"), str)
        if param_name in required:
            params.append(inspect.Parameter(param_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=annotation))
        else:
            default = prop.get("default", None)
            params.append(inspect.Parameter(param_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, default=default, annotation=annotation))

    async def _fn(*args, **kwargs) -> str:
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return await asyncio.to_thread(_call_tool_sync, name, dict(bound.arguments))

    sig = inspect.Signature(params, return_annotation=str)
    _fn.__name__      = name
    _fn.__doc__       = description
    _fn.__signature__ = sig
    return FunctionTool(func=_fn)


# ── Main ─────────────────────────────────────────────────────────────────────

async def run_agent():
    print(f"\nFashion Agent")
    print(f"  LLM       : openai/{OPENAI_MODEL}")
    print(f"  AgentCore : {AGENTCORE_ARN}\n")

    # Discover live tools from the MCP server
    print("Connecting to AgentCore MCP endpoint...", end=" ", flush=True)
    tools_resp = await asyncio.to_thread(_mcp_request, "tools/list")
    mcp_tools  = tools_resp.get("result", {}).get("tools", [])
    print(f"OK — {len(mcp_tools)} tools")
    for t in mcp_tools:
        print(f"  • {t['name']}: {t.get('description', '').strip().splitlines()[0]}")
    print()

    adk_tools = [make_adk_tool(t) for t in mcp_tools]

    agent = Agent(
        name="fashion_agent",
        model=LiteLlm(model=f"openai/{OPENAI_MODEL}"),
        instruction=(
            "You are a fashion assistant for Okkular. "
            "Use get_product_tags to retrieve product attributes, "
            "search_catalog to find items (required param: query, optional: gender/category/max_price), "
            "and generate_description to write marketing copy. "
            "Be concise and helpful."
        ),
        tools=adk_tools,
    )

    session_svc  = InMemorySessionService()
    runner       = Runner(agent=agent, app_name="fashion-agent", session_service=session_svc)
    user_session = await session_svc.create_session(app_name="fashion-agent", user_id="user-1")

    print("Agent ready. Type your query or 'exit'.")
    print("  'What tags does SKU-002 have?'")
    print("  'Find women dresses under $100'")
    print("  'Write a luxury description for SKU-004'\n")

    from google.genai.types import Content, Part

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not user_input or user_input.lower() in ("exit", "quit"):
            print("Bye!")
            break

        reply = ""
        async for event in runner.run_async(
            user_id="user-1",
            session_id=user_session.id,
            new_message=Content(parts=[Part(text=user_input)]),
        ):
            if event.is_final_response() and event.content:
                for part in event.content.parts:
                    if hasattr(part, "text") and part.text:
                        reply += part.text

        print(f"\nAgent: {reply}\n")


if __name__ == "__main__":
    asyncio.run(run_agent())
