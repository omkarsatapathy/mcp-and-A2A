#!/usr/bin/env bash
# deploy.sh — Build → ECR → Bedrock AgentCore (always updates existing runtime)
#
# Usage:
#   ./deploy.sh          # build + push + update AgentCore with tag "latest"
#   ./deploy.sh v1.2     # same with a custom image tag
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Colour helpers ─────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()    { echo -e "${CYAN}[INFO]${NC}  $*"; }
success() { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
die()     { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── Config (hardcoded — no exports needed) ────────────────────────────────────
AWS_PROFILE="personal-dev"
AWS_REGION="ap-south-1"
ACCOUNT_ID="177697910426"
ECR_REPO="research-agent"
AGENT_RUNTIME_NAME="ResearchAgentA2A"
ROLE_ARN="arn:aws:iam::177697910426:role/AgentCoreResearchAgentRole"

IMAGE_TAG="${1:-latest}"

ECR_REGISTRY="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
ECR_IMAGE="${ECR_REGISTRY}/${ECR_REPO}:${IMAGE_TAG}"

# Script lives inside a2a_protocol/ — build context is that same directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Alias aws to always use the right profile & region.
aws() { command aws --profile "${AWS_PROFILE}" --region "${AWS_REGION}" "$@"; }

# ── Pre-flight ─────────────────────────────────────────────────────────────────
info "Pre-flight checks..."
command -v docker >/dev/null 2>&1 || die "docker not found."
aws sts get-caller-identity --query Arn --output text >/dev/null \
  || die "AWS credentials not working for profile '${AWS_PROFILE}'. Run: aws configure --profile ${AWS_PROFILE}"
success "AWS identity OK."

echo ""
echo -e "${CYAN}════════════════════════════════════════${NC}"
echo -e "  Profile : ${AWS_PROFILE}"
echo -e "  Region  : ${AWS_REGION}"
echo -e "  Image   : ${ECR_IMAGE}"
echo -e "  Runtime : ${AGENT_RUNTIME_NAME}"
echo -e "${CYAN}════════════════════════════════════════${NC}"
echo ""

# ── Step 1: ECR login & repo ───────────────────────────────────────────────────
info "Step 1/3 — Authenticating with ECR..."
aws ecr get-login-password \
  | docker login --username AWS --password-stdin "${ECR_REGISTRY}"
success "ECR login successful."

if ! aws ecr describe-repositories --repository-names "${ECR_REPO}" --output text >/dev/null 2>&1; then
  warn "ECR repository '${ECR_REPO}' not found — creating it..."
  aws ecr create-repository \
    --repository-name "${ECR_REPO}" \
    --image-scanning-configuration scanOnPush=true \
    --output text >/dev/null
  success "ECR repository created."
fi

# ── Step 2: Build & push ───────────────────────────────────────────────────────
info "Step 2/3 — Building image (platform: linux/arm64)..."

HOST_ARCH="$(uname -m)"
if [[ "$HOST_ARCH" == "arm64" || "$HOST_ARCH" == "aarch64" ]]; then
  DOCKER_BUILD_CMD="docker build"
else
  if ! docker buildx inspect arm64-builder >/dev/null 2>&1; then
    warn "Creating cross-platform buildx builder for arm64..."
    docker buildx create --name arm64-builder --use --platform linux/arm64
    docker buildx inspect --bootstrap arm64-builder
  else
    docker buildx use arm64-builder
  fi
  DOCKER_BUILD_CMD="docker buildx build --load"
fi

$DOCKER_BUILD_CMD \
  --platform linux/arm64 \
  --tag "${ECR_REPO}:${IMAGE_TAG}" \
  --file "${SCRIPT_DIR}/Dockerfile" \
  "${SCRIPT_DIR}"

success "Build complete."
docker tag "${ECR_REPO}:${IMAGE_TAG}" "${ECR_IMAGE}"

info "  Pushing → ${ECR_IMAGE}"
docker push "${ECR_IMAGE}"
success "Image pushed."

# ── Step 3: Update Bedrock AgentCore runtime ───────────────────────────────────
# Note: `aws bedrock-agentcore` is not a standard CLI service — use boto3 directly.
info "Step 3/3 — Updating AgentCore runtime '${AGENT_RUNTIME_NAME}'..."

python3 - <<PYEOF
import sys, os, boto3

os.environ["AWS_PROFILE"]  = "${AWS_PROFILE}"
os.environ["AWS_DEFAULT_REGION"] = "${AWS_REGION}"

client = boto3.client("bedrock-agentcore-control", region_name="${AWS_REGION}")

# Find the runtime by name
runtimes = client.list_agent_runtimes().get("agentRuntimes", [])
match = next((r for r in runtimes if r["agentRuntimeName"] == "${AGENT_RUNTIME_NAME}"), None)

if not match:
    print(f"[ERROR] Runtime '${AGENT_RUNTIME_NAME}' not found in ${AWS_REGION}.", file=sys.stderr)
    sys.exit(1)

runtime_id = match["agentRuntimeId"]
print(f"  Found runtime ID: {runtime_id}")

client.update_agent_runtime(
    agentRuntimeId=runtime_id,
    agentRuntimeArtifact={"containerConfiguration": {"containerUri": "${ECR_IMAGE}"}},
    networkConfiguration={"networkMode": "PUBLIC"},
    roleArn="${ROLE_ARN}",
)
print(f"  AgentCore runtime updated successfully.")
PYEOF

success "AgentCore runtime updated."

# ── Done ───────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo -e "${GREEN}  Deployment complete!${NC}"
echo -e "  Image   : ${ECR_IMAGE}"
echo -e "  Runtime : ${AGENT_RUNTIME_NAME} (${AWS_REGION})"
echo -e "${GREEN}════════════════════════════════════════${NC}"
echo ""
info "Tail logs:"
echo "  aws --profile ${AWS_PROFILE} --region ${AWS_REGION} logs tail /aws/bedrock-agentcore/${AGENT_RUNTIME_NAME} --follow"
