import json

import httpx
import pytest
from pydantic import BaseModel

from marketplace_agent.llm.base import Message
from marketplace_agent.llm.providers import (
    LLMProviderError,
    OpenAICompatibleLLMClient,
)


def test_client_maps_chat_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert json.loads(request.content)["model"] == "test-model"

        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Ответ"}}],
                "model": "test-model",
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        )

    client = OpenAICompatibleLLMClient(
        api_key="test-key",
        base_url="https://provider.example/v1",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )

    response = client.chat(
        messages=[Message(role="user", content="Привет")],
        tools=None,
        response_schema=None,
        temperature=0.1,
        max_tokens=100,
    )

    assert response.content == "Ответ"
    assert response.prompt_tokens == 3
    assert response.completion_tokens == 2


def test_client_maps_http_error_to_safe_exception() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "Invalid key"}})

    client = OpenAICompatibleLLMClient(
        api_key="test-key",
        base_url="https://provider.example/v1",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LLMProviderError, match="401"):
        client.chat(
            messages=[Message(role="user", content="Привет")],
            tools=None,
            response_schema=None,
            temperature=0.1,
            max_tokens=100,
        )
def test_client_maps_timeout_to_safe_exception() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Timed out.", request=request)

    client = OpenAICompatibleLLMClient(
        api_key="test-key",
        base_url="https://provider.example/v1",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(LLMProviderError, match="request failed"):
        client.chat(
            messages=[Message(role="user", content="Привет")],
            tools=None,
            response_schema=None,
            temperature=0.1,
            max_tokens=100,
        )

def test_client_passes_configured_reasoning_effort() -> None:
    class Answer(BaseModel):
        status: str

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["reasoning_effort"] == "none"

        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"status":"ok"}'}}],
                "model": "test-model",
                "usage": {},
            },
        )

    client = OpenAICompatibleLLMClient(
        api_key="test-key",
        base_url="https://provider.example/v1",
        model="test-model",
        reasoning_effort="none",
        transport=httpx.MockTransport(handler),
    )

    client.chat(
        messages=[Message(role="user", content="Верни JSON.")],
        tools=None,
        response_schema=Answer,
        temperature=0.0,
        max_tokens=50,
    )

def test_client_retries_rate_limited_request(monkeypatch) -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            return httpx.Response(
                429,
                json={
                    "error": {
                        "message": "Please try again in 0ms.",
                    }
                },
            )

        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ОК"}}],
                "model": "test-model",
                "usage": {},
            },
        )

    monkeypatch.setattr(
        "marketplace_agent.llm.providers.sleep",
        lambda _: None,
        raising=False,
    )
    client = OpenAICompatibleLLMClient(
        api_key="test-key",
        base_url="https://provider.example/v1",
        model="test-model",
        transport=httpx.MockTransport(handler),
    )

    response = client.chat(
        messages=[Message(role="user", content="Привет")],
        tools=None,
        response_schema=None,
        temperature=0.0,
        max_tokens=50,
    )

    assert response.content == "ОК"
    assert call_count == 2