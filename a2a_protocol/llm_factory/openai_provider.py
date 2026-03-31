from openai import OpenAI
from a2a_protocol.llm_factory.base import BaseLLMProvider, LLMResponse


class OpenAIProvider(BaseLLMProvider):

    def __init__(self, api_key: str, model: str, temperature: float = 0.7):
        super().__init__(api_key, model, temperature)
        self._client = OpenAI(api_key=api_key)

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 2048) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_completion_tokens=max_tokens,
        )
        usage = resp.usage
        cost = (usage.prompt_tokens * 0.15 + usage.completion_tokens * 0.60) / 1_000_000
        return LLMResponse(
            content=resp.choices[0].message.content,
            model=resp.model,
            input_tokens=usage.prompt_tokens,
            output_tokens=usage.completion_tokens,
            cost_usd=round(cost, 8),
        )

    def get_provider_name(self) -> str:
        return "openai"
