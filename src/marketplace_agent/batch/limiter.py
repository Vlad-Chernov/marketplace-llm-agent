from collections.abc import Callable
from threading import Lock
from time import monotonic, sleep
from typing import Any

from pydantic import BaseModel

from marketplace_agent.llm.base import (
    LLMClient,
    LLMResponse,
    Message,
)


class RateLimiter:
    """Space requests evenly over one minute."""

    def __init__(
        self,
        requests_per_minute: int,
        *,
        clock: Callable[[], float] = monotonic,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be at least 1.")

        self._interval_seconds = 60 / requests_per_minute
        self._clock = clock
        self._sleeper = sleeper
        self._lock = Lock()
        self._next_allowed_at: float | None = None

    def acquire(self) -> None:
        """Wait until the next request may start."""

        with self._lock:
            now = self._clock()
            if self._next_allowed_at is None:
                delay = 0.0
                self._next_allowed_at = now + self._interval_seconds
            else:
                allowed_at = max(now, self._next_allowed_at)
                delay = allowed_at - now
                self._next_allowed_at = (
                    allowed_at + self._interval_seconds
                )

        if delay > 0:
            self._sleeper(delay)


class RateLimitedLLMClient:
    """Apply a shared request limit to an LLM client."""

    def __init__(self, client: LLMClient, limiter: RateLimiter) -> None:
        self._client = client
        self._limiter = limiter

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        response_schema: type[BaseModel] | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Limit and forward one chat request."""

        self._limiter.acquire()
        return self._client.chat(
            messages=messages,
            tools=tools,
            response_schema=response_schema,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Forward embedding requests without a chat limit."""

        return self._client.embed(texts)