import pytest
from pydantic import BaseModel

from marketplace_agent.llm.base import FakeLLMClient, LLMResponse, Message
from marketplace_agent.llm.structured import StructuredOutputError, chat_structured


class ExtractedRam(BaseModel):
    ram_gb: int

class RecordingLLMClient:
    def __init__(self) -> None:
        self.max_tokens: int | None = None

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, object]] | None,
        response_schema: type[BaseModel] | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        del messages, tools, response_schema, temperature
        self.max_tokens = max_tokens
        return LLMResponse(
            content='{"ram_gb": 16}',
            model="fake-model",
            prompt_tokens=1,
            completion_tokens=1,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        del texts
        return []


def test_retries_invalid_json_and_returns_valid_model() -> None:
    client = FakeLLMClient(
        chat_responses=[
            LLMResponse(
                content="это не JSON",
                model="fake-model",
                prompt_tokens=1,
                completion_tokens=1,
            ),
            LLMResponse(
                content='{"ram_gb": 16}',
                model="fake-model",
                prompt_tokens=1,
                completion_tokens=1,
            ),
        ]
    )

    result = chat_structured(
        client=client,
        messages=[Message(role="user", content="Извлеки RAM.")],
        response_schema=ExtractedRam,
        max_retries=1,
    )

    assert result.ram_gb == 16


def test_raises_controlled_error_after_retries_are_exhausted() -> None:
    client = FakeLLMClient(
        chat_responses=[
            LLMResponse(
                content='{"ram_gb": "много"}',
                model="fake-model",
                prompt_tokens=1,
                completion_tokens=1,
            )
        ]
    )

    with pytest.raises(StructuredOutputError):
        chat_structured(
            client=client,
            messages=[Message(role="user", content="Извлеки RAM.")],
            response_schema=ExtractedRam,
            max_retries=0,
        )

def test_requests_enough_tokens_for_structured_content() -> None:
    client = RecordingLLMClient()

    chat_structured(
        client=client,
        messages=[Message(role="user", content="Верни JSON.")],
        response_schema=ExtractedRam,
        max_retries=0,
    )

    assert client.max_tokens == 1024