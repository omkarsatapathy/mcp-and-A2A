"""
a2a_client.py — A2A v1.0 client for the Research Agent.

Discovers the agent via its AgentCard, then sends research queries
using JSON-RPC 2.0 (message/send and message/stream).

Usage:
    # Sync request (waits for full result)
    python a2a_client.py "Explain quantum computing in 200 words"

    # Streaming request
    python a2a_client.py --stream "Latest advances in renewable energy"

    # Custom endpoint
    python a2a_client.py --url http://my-server:8080 "My research query"

    # Discover agent capabilities
    python a2a_client.py --discover
"""

import argparse
import json
import sys
import uuid

import httpx

DEFAULT_URL = "http://localhost:8080"
TIMEOUT = 300.0  # 5 minutes — research pipeline can be slow


def discover(base_url: str) -> dict:
    """Fetch the AgentCard from the well-known endpoint."""
    url = f"{base_url}/.well-known/agent-card.json"
    resp = httpx.get(url, timeout=30.0)
    resp.raise_for_status()
    return resp.json()


def message_send(base_url: str, query: str, context_id: str | None = None) -> dict:
    """Send a synchronous message/send JSON-RPC request."""
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/send",
        "params": {
            "message": {
                "messageId": str(uuid.uuid4()),
                "role": "user",
                "parts": [{"kind": "text", "text": query}],
            }
        },
    }
    if context_id:
        payload["params"]["message"]["contextId"] = context_id

    resp = httpx.post(
        f"{base_url}/",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


def message_stream(base_url: str, query: str, context_id: str | None = None):
    """Send a message/stream JSON-RPC request and yield SSE events."""
    payload = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "message/stream",
        "params": {
            "message": {
                "messageId": str(uuid.uuid4()),
                "role": "user",
                "parts": [{"kind": "text", "text": query}],
            }
        },
    }
    if context_id:
        payload["params"]["message"]["contextId"] = context_id

    with httpx.stream(
        "POST",
        f"{base_url}/",
        json=payload,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        timeout=TIMEOUT,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line.startswith("data:"):
                data = line[len("data:"):].strip()
                if data:
                    yield json.loads(data)


def print_result(result: dict) -> None:
    """Pretty-print the task result from a message/send response."""
    task = result.get("result", result)
    status = task.get("status", {})
    state = status.get("state", "unknown")

    print(f"\n{'=' * 70}")
    print(f"  Task ID:    {task.get('id', 'N/A')}")
    print(f"  Context ID: {task.get('contextId', 'N/A')}")
    print(f"  State:      {state}")
    print(f"{'=' * 70}")

    if state == "failed":
        msg = status.get("message", {})
        parts = msg.get("parts", []) if isinstance(msg, dict) else []
        error_text = parts[0].get("text", "Unknown error") if parts else "Unknown error"
        print(f"\n  ERROR: {error_text}\n")
        return

    artifacts = task.get("artifacts", [])
    if artifacts:
        for artifact in artifacts:
            name = artifact.get("name", "unnamed")
            parts = artifact.get("parts", [])
            print(f"\n  Artifact: {name}")
            print(f"  {'-' * 66}")
            for part in parts:
                if part.get("kind") == "text":
                    text = part["text"]
                    word_count = len(text.split())
                    print(f"\n{text}")
                    print(f"\n  [{word_count} words]")
    else:
        print("\n  No artifacts returned.")

    print()


def print_agent_card(card: dict) -> None:
    """Pretty-print the AgentCard."""
    print(f"\n{'=' * 70}")
    print(f"  Agent: {card.get('name', 'Unknown')}")
    print(f"  Version: {card.get('version', 'N/A')}")
    print(f"  URL: {card.get('url', 'N/A')}")
    print(f"  Protocol: {card.get('protocolVersion', 'N/A')}")
    print(f"  Streaming: {card.get('capabilities', {}).get('streaming', False)}")
    print(f"{'=' * 70}")
    print(f"\n  Description: {card.get('description', 'N/A')}")

    for skill in card.get("skills", []):
        print(f"\n  Skill: {skill.get('name')} ({skill.get('id')})")
        print(f"    {skill.get('description', '')}")
        examples = skill.get("examples", [])
        if examples:
            print(f"    Examples:")
            for ex in examples:
                print(f"      - {ex}")
    print()


def main():
    parser = argparse.ArgumentParser(description="A2A v1.0 Research Agent Client")
    parser.add_argument("query", nargs="?", help="Research query to send")
    parser.add_argument("--url", default=DEFAULT_URL, help="Agent base URL")
    parser.add_argument("--stream", action="store_true", help="Use message/stream (SSE)")
    parser.add_argument("--discover", action="store_true", help="Fetch and display AgentCard")
    parser.add_argument("--context-id", default=None, help="Continue an existing conversation")
    args = parser.parse_args()

    if args.discover:
        card = discover(args.url)
        print_agent_card(card)
        return

    if not args.query:
        parser.error("Please provide a research query, or use --discover")

    print(f"Sending to {args.url} ...")
    print(f"Query: {args.query}")

    if args.stream:
        print("\n--- Streaming events ---")
        for event in message_stream(args.url, args.query, args.context_id):
            kind = event.get("kind", "")
            if kind == "status-update":
                state = event.get("status", {}).get("state", "")
                print(f"  [{state}]", end=" ", flush=True)
                msg = event.get("status", {}).get("message", {})
                parts = msg.get("parts", []) if isinstance(msg, dict) else []
                if parts:
                    print(parts[0].get("text", ""))
                else:
                    print()
            elif kind == "artifact-update":
                artifact = event.get("artifact", {})
                for part in artifact.get("parts", []):
                    if part.get("kind") == "text":
                        print(f"\n{part['text']}")
                        print(f"\n  [{len(part['text'].split())} words]")
            else:
                print(f"  Event: {kind}")
        print("\n--- Stream complete ---")
    else:
        result = message_send(args.url, args.query, args.context_id)
        if "error" in result:
            err = result["error"]
            print(f"\nJSON-RPC Error {err.get('code')}: {err.get('message')}")
            sys.exit(1)
        print_result(result)


if __name__ == "__main__":
    main()
