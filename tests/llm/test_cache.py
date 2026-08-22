from pathlib import Path
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

def test_counts_cache_hits_and_misses() -> None:
    inner_client = CountingLLMClient()
    client = CachedLLMClient(inner_client)
    messages = [Message(role="user", content="Привет")]

    client.chat(
        messages=messages,
        tools=None,
        response_schema=None,
        temperature=0.0,
        max_tokens=20,
    )
    client.chat(
        messages=messages,
        tools=None,
        response_schema=None,
        temperature=0.0,
        max_tokens=20,
    )

    assert client.cache_hits == 1
    assert client.cache_misses == 1

def test_cache_survives_new_client_instance(tmp_path: Path) -> None:
    cache_path = tmp_path / "llm-cache.json"
    messages = [Message(role="user", content="Привет")]

    first_inner_client = CountingLLMClient()
    first_client = CachedLLMClient(
        first_inner_client,
        cache_path=cache_path,
    )
    first_client.chat(
        messages=messages,
        tools=None,
        response_schema=None,
        temperature=0.0,
        max_tokens=20,
    )

    second_inner_client = CountingLLMClient()
    second_client = CachedLLMClient(
        second_inner_client,
        cache_path=cache_path,
    )
    second_client.chat(
        messages=messages,
        tools=None,
        response_schema=None,
        temperature=0.0,
        max_tokens=20,
    )

    assert first_inner_client.chat_calls == 1
    assert second_inner_client.chat_calls == 0
    assert second_client.cache_hits == 1