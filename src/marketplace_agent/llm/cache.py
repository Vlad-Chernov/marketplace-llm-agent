import hashlib
import json
from typing import Any

from pydantic import BaseModel

from marketplace_agent.llm.base import LLMClient, LLMResponse, Message


class CachedLLMClient:
    """Cache identical LLM requests in memory."""

    def __init__(self, inner_client: LLMClient) -> None:
        self.inner_client = inner_client
        self._chat_cache: dict[str, LLMResponse] = {}

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        response_schema: type[BaseModel] | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Return a cached response or call the inner client."""

        cache_key = self._create_chat_cache_key(
            messages=messages,
            tools=tools,
            response_schema=response_schema,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        if cache_key not in self._chat_cache:
            self._chat_cache[cache_key] = self.inner_client.chat(
                messages=messages,
                tools=tools,
                response_schema=response_schema,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        return self._chat_cache[cache_key]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Delegate embeddings without caching in the first MVP version."""

        return self.inner_client.embed(texts)

    @staticmethod
    def _create_chat_cache_key(
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        response_schema: type[BaseModel] | None,
        temperature: float,
        max_tokens: int,
    ) -> str:
        payload = {
            "messages": [message.model_dump() for message in messages],
            "tools": tools,
            "response_schema": (
                response_schema.__name__ if response_schema is not None else None
            ),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        serialized_payload = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
        )
        return hashlib.sha256(serialized_payload.encode()).hexdigest()