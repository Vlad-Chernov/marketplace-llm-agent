import pytest

from marketplace_agent.evals.llm_meter import MeteredLLMClient
from marketplace_agent.llm.base import FakeLLMClient, LLMResponse, Message
from marketplace_agent.llm.cache import CachedLLMClient


def test_accumulates_tokens_and_cost_for_llm_calls() -> None:
    client = MeteredLLMClient(
        FakeLLMClient(
            [
                LLMResponse(
                    content="Первый ответ",
                    model="fake-model",
                    prompt_tokens=100,
                    completion_tokens=20,
                ),
                LLMResponse(
                    content="Второй ответ",
                    model="fake-model",
                    prompt_tokens=50,
                    completion_tokens=10,
                ),
            ]
        ),
        input_price_per_million=1.0,
        output_price_per_million=2.0,
    )

    for _ in range(2):
        client.chat(
            messages=[Message(role="user", content="Вопрос")],
            tools=None,
            response_schema=None,
            temperature=0.0,
            max_tokens=100,
        )

    assert client.prompt_tokens == 150
    assert client.completion_tokens == 30
    assert client.cost_usd == 0.00021

def test_resets_accumulated_metrics() -> None:
    client = MeteredLLMClient(
        FakeLLMClient(
            [
                LLMResponse(
                    content="Ответ",
                    model="fake-model",
                    prompt_tokens=100,
                    completion_tokens=20,
                )
            ]
        ),
        input_price_per_million=1.0,
        output_price_per_million=2.0,
    )

    client.chat(
        messages=[Message(role="user", content="Вопрос")],
        tools=None,
        response_schema=None,
        temperature=0.0,
        max_tokens=100,
    )
    client.reset()

    assert client.prompt_tokens == 0
    assert client.completion_tokens == 0
    assert client.latency_ms == 0
    assert client.cost_usd == 0.0

def test_keeps_last_llm_response() -> None:
    expected_response = LLMResponse(
        content='{"ram_gb":"16"}',
        model="fake-model",
        prompt_tokens=10,
        completion_tokens=5,
    )
    client = MeteredLLMClient(
        FakeLLMClient([expected_response]),
        input_price_per_million=0.0,
        output_price_per_million=0.0,
    )

    client.chat(
        messages=[Message(role="user", content="Вопрос")],
        tools=None,
        response_schema=None,
        temperature=0.0,
        max_tokens=100,
    )

    assert client.last_response == expected_response

def test_does_not_bill_cached_llm_response() -> None:
    cached_client = CachedLLMClient(
        FakeLLMClient(
            [
                LLMResponse(
                    content="Ответ",
                    model="fake-model",
                    prompt_tokens=100,
                    completion_tokens=20,
                )
            ]
        )
    )
    client = MeteredLLMClient(
        cached_client,
        input_price_per_million=1.0,
        output_price_per_million=2.0,
    )
    messages = [Message(role="user", content="Вопрос")]

    client.chat(
        messages=messages,
        tools=None,
        response_schema=None,
        temperature=0.0,
        max_tokens=100,
    )
    client.chat(
        messages=messages,
        tools=None,
        response_schema=None,
        temperature=0.0,
        max_tokens=100,
    )

    assert client.prompt_tokens == 100
    assert client.completion_tokens == 20
    assert client.cost_usd == pytest.approx(0.00014)