"""Secrets service — resolves API keys from environment variables.

For local dev, set keys in your shell or .env file:
    export OPENAI_API_KEY="sk-..."
    export TAVILY_API_KEY="tvly-..."

For production, swap this module with the AWS Secrets Manager version
from the daily_news_automation_via_telegram project.
"""

import logging
import os

logger = logging.getLogger(__name__)


def get_secret(key_name: str) -> str:
    """Return the secret value for *key_name* from environment variables.

    Args:
        key_name: Environment variable name (e.g. 'OPENAI_API_KEY').

    Returns:
        The secret string value.

    Raises:
        ValueError: If the key is missing or empty.
    """
    value = os.environ.get(key_name, "").strip()
    if not value:
        raise ValueError(
            f"Environment variable '{key_name}' is missing or empty. "
            f"Set it via: export {key_name}='your-key-here'"
        )
    logger.debug("Resolved secret '%s'", key_name)
    return value
