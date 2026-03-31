"""Secrets service — resolves API keys.

Local dev:  reads from environment variables / .env file.
Production: reads from AWS Secrets Manager (set USE_AWS_SECRETS=true).

    export OPENAI_API_KEY="sk-..."
    export TAVILY_API_KEY="tvly-..."
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

_USE_AWS = os.getenv("USE_AWS_SECRETS", "").lower() in ("1", "true", "yes")
_AWS_SECRET_ID = os.getenv("AWS_SECRET_ID", "research-agent/api-keys")
_AWS_REGION = os.getenv("AWS_REGION", "ap-south-1")

# Cached AWS secret bundle (fetched once per process)
_aws_cache: dict[str, str] | None = None


def _get_aws_secrets() -> dict[str, str]:
    """Fetch all keys from AWS Secrets Manager (cached after first call)."""
    global _aws_cache
    if _aws_cache is not None:
        return _aws_cache

    import boto3

    client = boto3.client("secretsmanager", region_name=_AWS_REGION)
    response = client.get_secret_value(SecretId=_AWS_SECRET_ID)
    _aws_cache = json.loads(response["SecretString"])
    logger.info("Loaded secrets from AWS Secrets Manager (%s)", _AWS_SECRET_ID)
    return _aws_cache


def get_secret(key_name: str) -> str:
    """Return the secret value for *key_name*.

    Reads from AWS Secrets Manager when USE_AWS_SECRETS is set,
    otherwise falls back to environment variables.

    Args:
        key_name: Secret key (e.g. 'OPENAI_API_KEY').

    Returns:
        The secret string value.

    Raises:
        ValueError: If the key is missing or empty.
    """
    if _USE_AWS:
        secrets = _get_aws_secrets()
        value = secrets.get(key_name, "").strip()
        if not value:
            raise ValueError(
                f"Key '{key_name}' not found in AWS secret '{_AWS_SECRET_ID}'"
            )
        logger.debug("Resolved secret '%s' from AWS Secrets Manager", key_name)
        return value

    # Local / env-var fallback
    value = os.environ.get(key_name, "").strip()
    if not value:
        raise ValueError(
            f"Environment variable '{key_name}' is missing or empty. "
            f"Set it via: export {key_name}='your-key-here'"
        )
    logger.debug("Resolved secret '%s' from environment", key_name)
    return value
