import json

import httpx
import pytest

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