"""
run_remote.py — Invoke the deployed Bedrock AgentCore runtime directly.

Sends the query to the live ResearchAgentA2A runtime running on AWS
Bedrock AgentCore (ap-south-1) instead of running the pipeline locally.

The runtime receives the payload at its /invocations endpoint:
    {"input": {"prompt": "..."}}

Usage:
    python -m a2a_protocol.scripts.run_remote
    python -m a2a_protocol.scripts.run_remote <<< "Your query here"

Environment (required):
    AWS_PROFILE           — AWS credentials profile (default: personal-dev)
    AWS_REGION            — AWS region (default: ap-south-1)
    AGENTCORE_RUNTIME_NAME — AgentCore runtime name (default: ResearchAgentA2A)
    AGENTCORE_RUNTIME_ARN  — Optional: skip list_agent_runtimes lookup if set
"""

import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

AWS_PROFILE            = os.getenv("AWS_PROFILE", "personal-dev")
AWS_REGION             = os.getenv("AWS_REGION", "ap-south-1")
RUNTIME_NAME           = os.getenv("AGENTCORE_RUNTIME_NAME", "ResearchAgentA2A")
RUNTIME_ARN_OVERRIDE   = os.getenv("AGENTCORE_RUNTIME_ARN", "")


def _resolve_runtime_arn() -> str:
    """Return the AgentCore runtime ARN — use env override or look it up."""
    if RUNTIME_ARN_OVERRIDE:
        logger.info("Using runtime ARN from env: %s", RUNTIME_ARN_OVERRIDE)
        return RUNTIME_ARN_OVERRIDE

    import boto3

    logger.info(
        "Looking up runtime '%s' in %s (profile: %s)…",
        RUNTIME_NAME, AWS_REGION, AWS_PROFILE,
    )
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    control = session.client("bedrock-agentcore-control")
    runtimes = control.list_agent_runtimes().get("agentRuntimes", [])
    match = next(
        (r for r in runtimes if r["agentRuntimeName"] == RUNTIME_NAME), None
    )
    if not match:
        raise RuntimeError(
            f"AgentCore runtime '{RUNTIME_NAME}' not found in {AWS_REGION}. "
            "Deploy it first with: ./a2a_protocol/deploy.sh"
        )
    arn = match["agentRuntimeArn"]
    logger.info("Resolved runtime ARN: %s", arn)
    return arn


def invoke_remote(query: str) -> str:
    """Send query to the Bedrock AgentCore runtime and return the output text."""
    import boto3

    runtime_arn = _resolve_runtime_arn()
    payload = json.dumps({"input": {"prompt": query}}).encode("utf-8")

    logger.info("Invoking AgentCore runtime '%s'…", RUNTIME_NAME)

    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    data_client = session.client("bedrock-agentcore")

    response = data_client.invoke_agent_runtime(
        agentRuntimeArn=runtime_arn,
        qualifier="DEFAULT",
        payload=payload,
        contentType="application/json",
        accept="application/json",
    )

    raw = response["response"].read()
    logger.info("Response status: %s, size: %d bytes", response.get("statusCode"), len(raw))
    body = json.loads(raw)
    return body.get("output", {}).get("message", str(body))


if __name__ == "__main__":
    query = input("Enter your research query: ").strip()
    if not query:
        print("No query provided.", file=sys.stderr)
        sys.exit(1)

    output = invoke_remote(query)

    print("\n" + "=" * 80)
    print(f"REMOTE RESEARCH OUTPUT ({len(output.split())} words)")
    print("=" * 80)
    print(output)
