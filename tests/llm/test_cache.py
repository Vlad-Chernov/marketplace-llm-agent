from typing import Any

from pydantic import BaseModel

from marketplace_agent.llm.base import LLMResponse, Message
from marketplace_agent.llm.cache import CachedLLMClient


class CountingLLMClient:
    def __init__(self) -> None:
        self.chat_calls = 0

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        response_schema: type[BaseModel] | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        self.chat_calls += 1
        return LLMResponse(
            content="ОК",
            model="counting-model",
            prompt_tokens=1,
            completion_tokens=1,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] for _ in texts]


def test_cache_prevents_repeated_chat_request() -> None:
    inner_client = CountingLLMClient()
    client = CachedLLMClient(inner_client)
    messages = [Message(role="user", content="Привет")]

    first_response = client.chat(
        messages=messages,
        tools=None,
        response_schema=None,
        temperature=0.0,
        max_tokens=20,
    )
    second_response = client.chat(
        messages=messages,
        tools=None,
        response_schema=None,
        temperature=0.0,
        max_tokens=20,
    )

    assert first_response == second_response
    assert inner_client.chat_calls == 1