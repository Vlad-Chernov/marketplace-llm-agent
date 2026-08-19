import re
from time import sleep
from typing import Any

import httpx
from pydantic import BaseModel

from marketplace_agent.llm.base import LLMResponse, Message

RATE_LIMIT_DELAY_PATTERN = re.compile(r"in\s+([0-9.]+)(ms|s)")
class LLMProviderError(RuntimeError):
    """Represents a safe, provider-independent HTTP error."""


class OpenAICompatibleLLMClient:
    """Call an OpenAI-compatible chat-completions API."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        reasoning_effort: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.transport = transport

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        response_schema: type[BaseModel] | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Send a chat completion request and normalize its response."""

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [message.model_dump() for message in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if tools is not None:
            payload["tools"] = tools
        if response_schema is not None:
           payload["response_format"] = {"type": "json_object"}

        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort

        data = self._post("/chat/completions", payload)
        usage = data.get("usage", {})
        content = data["choices"][0]["message"].get("content") or ""

        return LLMResponse(
            content=content,
            model=data.get("model", self.model),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Request embedding vectors from the provider."""

        data = self._post(
            "/embeddings",
            {
                "model": self.model,
                "input": texts,
            },
        )
        return [item["embedding"] for item in data["data"]]

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        response: httpx.Response | None = None

        for attempt in range(3):
            try:
                with httpx.Client(
                    base_url=self.base_url,
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    timeout=30.0,
                    transport=self.transport,
                ) as client:
                    response = client.post(path, json=payload)
            except httpx.HTTPError as error:
                raise LLMProviderError("LLM request failed.") from error

            if response.status_code != 429 or attempt == 2:
                break

            error_message = response.json().get("error", {}).get("message", "")
            delay_match = RATE_LIMIT_DELAY_PATTERN.search(error_message)
            if delay_match is None:
                sleep(1.0)
                continue

            value, unit = delay_match.groups()
            delay_seconds = float(value) / 1000 if unit == "ms" else float(value)
            sleep(max(delay_seconds, 0.1))

        assert response is not None

        if response.status_code >= 400:
            error_message = response.json().get("error", {}).get(
                "message",
                "Unknown provider error.",
            )
            raise LLMProviderError(
                f"LLM provider returned HTTP {response.status_code}: "
                f"{error_message}"
            )

        return response.json()