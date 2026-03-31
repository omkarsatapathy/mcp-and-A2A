from abc import ABC, abstractmethod
from pydantic import BaseModel


class LLMResponse(BaseModel):
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class BaseLLMProvider(ABC):
    """Abstract interface for all LLM providers."""

    def __init__(self, api_key: str, model: str, temperature: float = 0.7):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature

    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 2048,
    ) -> LLMResponse:
        """Generate a response. Must be implemented by all providers."""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Return the provider name for logging."""
        pass
