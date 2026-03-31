#!/usr/bin/env bash
# ---------------------------------------------------------------
# Deploy fashion-mcp-server to AWS AgentCore Runtime (ap-southeast-2)
# Run this from inside fashion-mcp-server/
# ---------------------------------------------------------------
set -e

echo "==> Installing AgentCore deployment toolkit..."
pip install bedrock-agentcore-starter-toolkit -q

echo ""
echo "==> Configuring AgentCore Runtime deployment..."
echo "    This will ask a few questions — answer as follows:"
echo "      Protocol:    MCP"
echo "      Region:      ap-southeast-2"
echo "      Entrypoint:  fashion_tools.py"
echo ""
agentcore configure -e fashion_tools.py --protocol MCP --region ap-southeast-2

echo ""
echo "==> Launching to AgentCore Runtime..."
echo "    (CodeBuild will build your container + push to ECR — ~2 min)"
agentcore launch

echo ""
echo "==> Deployment complete! Run the following to get your endpoint URL:"
echo "    agentcore status"
echo ""
echo "    Copy the Runtime ID and Cognito details into local_agent/.env"
