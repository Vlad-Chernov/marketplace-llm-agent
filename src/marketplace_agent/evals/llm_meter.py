from time import perf_counter
from typing import Any

from pydantic import BaseModel

from marketplace_agent.llm.base import LLMClient, LLMResponse, Message


class MeteredLLMClient:
    """Wrap an LLM client and accumulate evaluation usage metrics."""

    def __init__(
        self,
        inner_client: LLMClient,
        input_price_per_million: float,
        output_price_per_million: float,
    ) -> None:
        self._inner_client = inner_client
        self._input_price_per_million = input_price_per_million
        self._output_price_per_million = output_price_per_million
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.latency_ms = 0
        self.cost_usd = 0.0


    def reset(self) -> None:
        """Clear metrics before running the next evaluation case."""

        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.latency_ms = 0
        self.cost_usd = 0.0

        
    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        response_schema: type[BaseModel] | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Call the inner client and record usage."""

        started_at = perf_counter()
        response = self._inner_client.chat(
            messages=messages,
            tools=tools,
            response_schema=response_schema,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.prompt_tokens += response.prompt_tokens
        self.completion_tokens += response.completion_tokens
        self.latency_ms += round((perf_counter() - started_at) * 1000)
        self.cost_usd += (
            response.prompt_tokens
            / 1_000_000
            * self._input_price_per_million
            + response.completion_tokens
            / 1_000_000
            * self._output_price_per_million
        )
        return response

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Delegate embedding requests without changing their metrics."""

        return self._inner_client.embed(texts)