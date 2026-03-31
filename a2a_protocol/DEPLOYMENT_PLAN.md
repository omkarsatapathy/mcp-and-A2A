# Deployment Plan: Research Agent → AWS Bedrock AgentCore with A2A Protocol (v1.0)

## Overview

Deploy the LangGraph research pipeline as an **A2A v1.0-compliant agent** hosted on **AWS Bedrock AgentCore**, exposed as a discoverable, interoperable agent service using the official A2A Python SDK.

> **Protocol version**: A2A v1.0 (Linux Foundation) — uses the official `a2a` Python SDK (`a2aproject/a2a-python`), `A2AStarletteApplication`, JSON-RPC 2.0 transport over HTTP/SSE, and the `/.well-known/agent-card.json` discovery endpoint.

---

## Architecture

```
Client / Orchestrator
        │  A2A JSON-RPC 2.0 (HTTP + SSE)
        │  methods: message/send, message/stream
        ▼
┌──────────────────────────────────────────┐
│  AWS Bedrock AgentCore                   │
│  ┌──────────────────────────────────┐    │
│  │  A2AStarletteApplication (ASGI)  │    │
│  │  ├── GET /.well-known/           │    │
│  │  │       agent-card.json         │    │
│  │  └── POST /  (JSON-RPC 2.0)      │    │
│  │       ├── message/send           │    │
│  │       └── message/stream (SSE)   │    │
│  │            └── ResearchAgentExecutor  │
│  │                 └── LangGraph Pipeline│
│  │                     input_parser →    │
│  │                     research →        │
│  │                     dedup →           │
│  │                     summarizer        │
│  └──────────────────────────────────┘    │
└──────────────────────────────────────────┘
        │
   AWS Secrets Manager (API keys)
   Amazon ECR (container image)
   CloudWatch (logs/metrics)
```

---

## Phase 1 — A2A Protocol Wrapper (v1.0 SDK Pattern)

### 1.1 Install A2A SDK

```bash
# The correct package name for A2A v1.0 is 'a2a' (NOT 'a2a-sdk')
pip install a2a uvicorn boto3
```

### 1.2 AgentCard Definition (in-code, not a JSON file)

Per v1.0 spec, the `AgentCard` is built in Python using `a2a.types` and served
automatically by `A2AStarletteApplication` at `/.well-known/agent-card.json`
(RFC 8615 well-known URI). No separate `agent_card.json` file is needed.

Key v1.0 fields:
- `supported_interfaces` (list of `AgentInterface`) — replaces the old flat `"url"` field
- `default_input_modes` / `default_output_modes` — MIME types (e.g. `"text/plain"`)
- `capabilities` — `AgentCapabilities(streaming=True)`
- Skills use `tags` and `examples` in addition to `inputModes`/`outputModes`

### 1.3 Agent Executor (`agent_executor.py`)

The v1.0 SDK pattern requires implementing `AgentExecutor` (NOT a raw FastAPI handler).
The executor receives a `RequestContext` and writes events to an `EventQueue`.

```python
# agent_executor.py
import asyncio
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message, new_task, new_text_artifact
from a2a.types import (
    TaskArtifactUpdateEvent,
    TaskState,
    TaskStatus,
    TaskStatusUpdateEvent,
)
from agent import run_research_agent  # existing LangGraph pipeline


class ResearchAgentExecutor(AgentExecutor):
    """Bridges the A2A v1.0 protocol to the LangGraph research pipeline."""

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
                status=TaskStatus(
                    state=TaskState.TASK_STATE_WORKING,
                    message=new_agent_text_message("Researching…"),
                ),
            )
        )

        # 3. Extract user query from the first text Part
        query = context.message.parts[0].root.text  # Part uses oneof: text/raw/url/data

        # 4. Run the LangGraph pipeline (blocking → offload to thread)
        result = await asyncio.get_event_loop().run_in_executor(
            None, run_research_agent, query
        )

        # 5. Return the research summary as an Artifact
        await event_queue.enqueue_event(
            TaskArtifactUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                artifact=new_text_artifact(name="research_summary", text=result),
            )
        )

        # 6. Mark complete
        await event_queue.enqueue_event(
            TaskStatusUpdateEvent(
                task_id=context.task_id,
                context_id=context.context_id,
                status=TaskStatus(state=TaskState.TASK_STATE_COMPLETED),
            )
        )

    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> None:
        raise Exception("cancel not supported")
```

### 1.4 A2A Server Entry Point (`a2a_server.py`)

```python
# a2a_server.py
import uvicorn
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
)
from agent_executor import ResearchAgentExecutor

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

agent_card = AgentCard(
    name="Research Agent",
    description="LangGraph research pipeline: search → scrape → dedup → summarize.",
    version="1.0.0",
    default_input_modes=["text/plain"],
    default_output_modes=["text/plain"],
    capabilities=AgentCapabilities(streaming=True),
    supported_interfaces=[
        AgentInterface(
            protocol_binding="JSONRPC",
            url="https://<agentcore-endpoint>",
        )
    ],
    skills=[skill],
)

request_handler = DefaultRequestHandler(
    agent_executor=ResearchAgentExecutor(),
    task_store=InMemoryTaskStore(),
)

server = A2AStarletteApplication(
    agent_card=agent_card,
    http_handler=request_handler,
)

if __name__ == "__main__":
    uvicorn.run(server.build(), host="0.0.0.0", port=8080)
```

> **Why this pattern?**  
> `A2AStarletteApplication` automatically exposes `GET /.well-known/agent-card.json`
> and routes all JSON-RPC 2.0 calls (`message/send`, `message/stream`, `tasks/get`, etc.)
> through `DefaultRequestHandler` → `ResearchAgentExecutor`. You do not write
> raw SSE generators or custom FastAPI routes.

---

## Phase 2 — Secrets Migration

Replace `.env` file with **AWS Secrets Manager**:

```python
# services/secrets_service.py — production swap
import boto3, json

def get_secret(key_name: str) -> str:
    client = boto3.client("secretsmanager", region_name="us-east-1")
    secret = client.get_secret_value(SecretId="research-agent/api-keys")
    return json.loads(secret["SecretString"])[key_name]
```

**Create secrets in AWS:**
```bash
aws secretsmanager create-secret \
  --name research-agent/api-keys \
  --secret-string '{"OPENAI_API_KEY":"...","TAVILY_API_KEY":"..."}'
```

---

## Phase 3 — Containerization

### 3.1 `Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Install app deps + A2A v1.0 SDK (package name is 'a2a', not 'a2a-sdk')
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install a2a uvicorn boto3

# Pre-download ML models at build time (avoid cold-start latency)
RUN python -c "
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer('all-MiniLM-L6-v2')
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
"

COPY . .

EXPOSE 8080
CMD ["python", "a2a_server.py"]
```

### 3.2 `requirements.txt` additions

```
a2a>=1.0.0
uvicorn>=0.30.0
boto3>=1.34.0
```

> **Note**: Do NOT use `a2a-sdk` — that is the old pre-v1.0 package name. The correct
> PyPI package for A2A v1.0 is `a2a` (from `a2aproject/a2a-python`).

### 3.3 Build & Push to ECR

```bash
# Create ECR repo
aws ecr create-repository --repository-name research-agent

# Authenticate and push
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin <account_id>.dkr.ecr.us-east-1.amazonaws.com

docker build -t research-agent .
docker tag research-agent:latest <account_id>.dkr.ecr.us-east-1.amazonaws.com/research-agent:latest
docker push <account_id>.dkr.ecr.us-east-1.amazonaws.com/research-agent:latest
```

---

## Phase 4 — Bedrock AgentCore Deployment

### 4.1 IAM Role

```json
{
  "Statement": [
    { "Effect": "Allow", "Action": ["secretsmanager:GetSecretValue"], "Resource": "arn:aws:secretsmanager:*:*:secret:research-agent/*" },
    { "Effect": "Allow", "Action": ["logs:CreateLogGroup", "logs:PutLogEvents", "logs:CreateLogDelivery"], "Resource": "*" },
    { "Effect": "Allow", "Action": ["bedrock:InvokeModel"], "Resource": "*" }
  ]
}
```

### 4.2 Deploy Agent

```bash
aws bedrock-agentcore create-agent-runtime \
  --agent-runtime-name "research-agent" \
  --description "LangGraph research pipeline with A2A v1.0 protocol" \
  --agent-runtime-artifact '{"containerConfiguration": {"containerUri": "<ecr_uri>:latest"}}' \
  --network-configuration '{"networkMode": "PUBLIC"}' \
  --role-arn arn:aws:iam::<account_id>:role/research-agent-role
```

### 4.3 Environment Variables (non-secret)

Set in AgentCore runtime config:
```
LOG_LEVEL=INFO
```

---

## Phase 5 — Validation

```bash
# 1. Discover agent — correct path is /.well-known/agent-card.json (RFC 8615)
curl https://<agentcore-endpoint>/.well-known/agent-card.json

# 2. Submit a research task using JSON-RPC 2.0 (message/send method)
curl -X POST https://<agentcore-endpoint>/ \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "req-001",
    "method": "message/send",
    "params": {
      "message": {
        "role": "user",
        "messageId": "msg-001",
        "parts": [{"text": "Explain quantum computing in 300 words"}]
      }
    }
  }'

# 3. Stream a research task using JSON-RPC 2.0 (message/stream method → SSE)
curl -X POST https://<agentcore-endpoint>/ \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{
    "jsonrpc": "2.0",
    "id": "req-002",
    "method": "message/stream",
    "params": {
      "message": {
        "role": "user",
        "messageId": "msg-002",
        "parts": [{"text": "Summarize recent advances in fusion energy"}]
      }
    }
  }'

# 4. Verify CloudWatch logs
aws logs tail /aws/bedrock-agentcore/research-agent --follow
```

> **Task lifecycle events** returned over SSE:  
> `Task (submitted)` → `TaskStatusUpdateEvent (WORKING)` → `TaskArtifactUpdateEvent (artifact)` → `TaskStatusUpdateEvent (COMPLETED)`

---

## Key Configuration Changes

| Item | Local | Production |
|---|---|---|
| Secrets | `.env` file | AWS Secrets Manager |
| Models | HuggingFace download on startup | Baked into Docker image |
| Server | `python run.py` | `python a2a_server.py` (A2AStarletteApplication) |
| Port | — | 8080 |
| A2A discovery endpoint | None | `GET /.well-known/agent-card.json` |
| A2A task endpoint | None | `POST /` (JSON-RPC 2.0: `message/send`, `message/stream`) |
| A2A SDK package | — | `a2a` (NOT `a2a-sdk`) |

---

## Deployment Checklist

- [ ] Implement `agent_executor.py` — `ResearchAgentExecutor` extending `AgentExecutor`
- [ ] Implement `a2a_server.py` — `A2AStarletteApplication` with `AgentCard` + `DefaultRequestHandler`
- [ ] Swap `secrets_service.py` to use AWS Secrets Manager
- [ ] Add `a2a>=1.0.0`, `uvicorn>=0.30.0`, `boto3>=1.34.0` to `requirements.txt`
- [ ] Write `Dockerfile` with ML model pre-bake
- [ ] Create ECR repository and push image
- [ ] Create IAM role with Secrets Manager + CloudWatch permissions
- [ ] Deploy to Bedrock AgentCore via CLI or Console
- [ ] Validate `GET /.well-known/agent-card.json` returns the AgentCard
- [ ] Validate `POST /` with `message/send` returns a completed Task
- [ ] Validate `POST /` with `message/stream` streams SSE events in correct order
- [ ] Confirm CloudWatch logs streaming

---

## Notes

- **Cold start**: ML models (`all-MiniLM-L6-v2`, `cross-encoder`) are ~400MB — bake into Docker image to eliminate startup latency.
- **Concurrency**: The existing `ThreadPoolExecutor` scraping is CPU-bound; `run_in_executor` in the async executor offloads it cleanly. Scale via AgentCore replicas.
- **OpenAI vs Bedrock models**: Agent currently uses OpenAI. To use Bedrock models natively, swap `OpenAIProvider` with a `BedrockProvider` in `llm_factory/`.
- **Task immutability**: Per A2A v1.0 spec, once a task reaches `completed/failed/canceled`, it cannot be restarted. Follow-up queries must use a new `message/send` call (optionally with the same `contextId` for continuity).
- **agent-card.json vs agent.json**: The spec mandates `/.well-known/agent-card.json` (RFC 8615). The old `/.well-known/agent.json` path is non-compliant and will fail discovery.
- **JSON-RPC 2.0**: All requests must include `"jsonrpc": "2.0"`, `"id"`, and `"method"`. Direct `POST /` with a raw Task body is not A2A v1.0 compliant.
