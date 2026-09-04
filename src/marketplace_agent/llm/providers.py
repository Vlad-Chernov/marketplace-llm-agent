import re
from time import sleep, time
from typing import Any
from uuid import uuid4

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
            payload["response_format"] = self._response_format(response_schema)

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

    def _response_format(
        self,
        response_schema: type[BaseModel],
    ) -> dict[str, Any]:
        return {"type": "json_object"}

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
                if attempt == 2:
                    raise LLMProviderError("LLM request failed.") from error

                sleep(1.0)
                continue

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

class GigaChatLLMClient(OpenAICompatibleLLMClient):
    """Call GigaChat after obtaining an OAuth access token."""

    def __init__(
        self,
        authorization_key: str,
        model: str,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__(
            api_key="",
            base_url="https://api.giga.chat/v1",
            model=model,
            transport=transport,
        )
        self.authorization_key = authorization_key
        self._access_token = ""
        self._token_expires_at = 0.0

    def _response_format(
        self,
        response_schema: type[BaseModel],
    ) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "schema": response_schema.model_json_schema(),
            "strict": True,
        }

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        self.api_key = self._get_access_token()
        return super()._post(path, payload)

    def _get_access_token(self) -> str:
        if self._access_token and time() < self._token_expires_at:
            return self._access_token

        try:
            with httpx.Client(
                base_url="https://ngw.devices.sberbank.ru:9443",
                headers={
                    "Authorization": f"Basic {self.authorization_key}",
                    "RqUID": str(uuid4()),
                },
                timeout=30.0,
                transport=self.transport,
            ) as client:
                response = client.post(
                    "/api/v2/oauth",
                    data={"scope": "GIGACHAT_API_PERS"},
                )
        except httpx.HTTPError as error:
            raise LLMProviderError("GigaChat authorization failed.") from error

        if response.status_code >= 400:
            raise LLMProviderError(
                f"GigaChat authorization returned HTTP {response.status_code}."
            )

        payload = response.json()
        self._access_token = payload["access_token"]
        self._token_expires_at = float(payload["expires_at"]) - 60
        return self._access_token