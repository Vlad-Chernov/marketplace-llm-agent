from typing import Any

from pydantic import BaseModel

from marketplace_agent.llm.base import LLMResponse, Message
from marketplace_agent.llm.progress import ProgressLLMClient


class FixedResponseClient:
    model = "fake-model"

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        response_schema: type[BaseModel] | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        del messages, tools, response_schema, temperature, max_tokens
        return LLMResponse(
            content="OK",
            model="fake-model",
            prompt_tokens=1,
            completion_tokens=1,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] for _ in texts]


def test_progress_client_reports_llm_request_duration() -> None:
    messages: list[str] = []
    clock_values = iter([10.0, 12.4])
    client = ProgressLLMClient(
        FixedResponseClient(),
        output=messages.append,
        clock=lambda: next(clock_values),
    )

    response = client.chat(
        messages=[Message(role="user", content="Привет")],
        tools=None,
        response_schema=None,
        temperature=0.0,
        max_tokens=20,
    )

    assert response.content == "OK"
    assert messages == [
        "LLM: запрос к fake-model...",
        "LLM: готово за 2.4 с.",
    ]
