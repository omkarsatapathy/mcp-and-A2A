from a2a_protocol.config import LLM_CONFIG
from a2a_protocol.llm_factory.base import BaseLLMProvider
from a2a_protocol.services.secrets_service import get_secret


def _get_provider_class(provider: str):
    """Lazily import provider classes."""
    if provider == "openai":
        from a2a_protocol.llm_factory.openai_provider import OpenAIProvider
        return OpenAIProvider
    raise ValueError(f"Unknown provider: {provider}")


def get_llm(role: str = "default") -> BaseLLMProvider:
    """
    Returns a configured LLM provider for the given role.
      role='research'   -> fast/cheap model for query generation
      role='summarizer' -> higher quality for final summary
    """
    cfg = LLM_CONFIG.get(role, LLM_CONFIG["default"])
    provider_cls = _get_provider_class(cfg["provider"])
    api_key = get_secret(cfg["secret_key_name"])
    return provider_cls(
        api_key=api_key,
        model=cfg["model"],
        temperature=cfg.get("temperature", 0.7),
    )
