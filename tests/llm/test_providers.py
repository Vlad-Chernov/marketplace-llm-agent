import json

import httpx
import pytest
from pydantic import BaseModel

from marketplace_agent.llm import providers
from marketplace_agent.llm.base import Message
from marketplace_agent.llm.providers import (
    GigaChatLLMClient,
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

    with pytest.raises(
        LLMProviderError,
        match="LLM request failed: ReadTimeout\\.",
    ):
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

def test_gigachat_client_obtains_token_and_maps_chat_response() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)

        if request.url.host == "ngw.devices.sberbank.ru":
            assert request.url.path == "/api/v2/oauth"
            assert request.headers["authorization"] == "Basic authorization-key"
            assert request.content == b"scope=GIGACHAT_API_PERS"
            return httpx.Response(
                200,
                json={
                    "access_token": "access-token",
                    "expires_at": 2_000_000_000,
                },
            )

        assert request.url == "https://api.giga.chat/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer access-token"
        assert json.loads(request.content)["model"] == "GigaChat-2-Pro"

        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "Ответ"}}],
                "model": "GigaChat-2-Pro",
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
        )

    client = GigaChatLLMClient(
        authorization_key="authorization-key",
        model="GigaChat-2-Pro",
        transport=httpx.MockTransport(handler),
    )

    response = client.chat(
        messages=[Message(role="user", content="Привет")],
        tools=None,
        response_schema=None,
        temperature=0.0,
        max_tokens=100,
    )

    assert response.content == "Ответ"
    assert response.prompt_tokens == 3
    assert response.completion_tokens == 2
    assert len(requests) == 2

def test_gigachat_client_uses_json_schema_for_structured_output() -> None:
    class Answer(BaseModel):
        status: str

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "ngw.devices.sberbank.ru":
            return httpx.Response(
                200,
                json={
                    "access_token": "access-token",
                    "expires_at": 2_000_000_000,
                },
            )

        payload = json.loads(request.content)

        assert payload["response_format"] == {
            "type": "json_schema",
            "schema": Answer.model_json_schema(),
            "strict": True,
        }

        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"status":"ok"}'}}],
                "model": "GigaChat-2-Pro",
                "usage": {},
            },
        )

    client = GigaChatLLMClient(
        authorization_key="authorization-key",
        model="GigaChat-2-Pro",
        transport=httpx.MockTransport(handler),
    )

    client.chat(
        messages=[Message(role="user", content="Верни JSON.")],
        tools=None,
        response_schema=Answer,
        temperature=0.0,
        max_tokens=50,
    )

def test_client_retries_transient_request_error(monkeypatch) -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            raise httpx.ReadTimeout("Timed out.", request=request)

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

def test_client_retries_transient_server_error_with_jitter(
    monkeypatch,
) -> None:
    call_count = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            return httpx.Response(
                503,
                json={"error": {"message": "Service unavailable"}},
            )

        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ОК"}}],
                "model": "test-model",
                "usage": {},
            },
        )

    monkeypatch.setattr(providers, "sleep", delays.append)
    monkeypatch.setattr(
        providers,
        "uniform",
        lambda start, stop: stop / 2,
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
    assert delays == [0.5]

def test_client_honors_retry_after_header(monkeypatch) -> None:
    call_count = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            return httpx.Response(
                429,
                headers={"Retry-After": "3"},
                json={"error": {"message": "Too many requests"}},
            )

        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ОК"}}],
                "model": "test-model",
                "usage": {},
            },
        )

    monkeypatch.setattr(providers, "sleep", delays.append)
    monkeypatch.setattr(providers, "uniform", lambda *_: 0.5)
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
    assert delays == [3.0]

def test_gigachat_retries_transient_oauth_error(monkeypatch) -> None:
    oauth_call_count = 0
    delays: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal oauth_call_count

        if request.url.host == "ngw.devices.sberbank.ru":
            oauth_call_count += 1
            if oauth_call_count == 1:
                return httpx.Response(503, json={"error": {}})
            return httpx.Response(
                200,
                json={
                    "access_token": "access-token",
                    "expires_at": 2_000_000_000,
                },
            )

        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "ОК"}}],
                "model": "GigaChat-2-Pro",
                "usage": {},
            },
        )

    monkeypatch.setattr(providers, "sleep", delays.append)
    monkeypatch.setattr(providers, "uniform", lambda *_: 0.5)
    client = GigaChatLLMClient(
        authorization_key="authorization-key",
        model="GigaChat-2-Pro",
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
    assert oauth_call_count == 2
    assert delays == [0.5]

def test_client_does_not_retry_non_transient_error() -> None:
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        return httpx.Response(
            401,
            json={"error": {"message": "Invalid key"}},
        )

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
            temperature=0.0,
            max_tokens=50,
        )

    assert call_count == 1
