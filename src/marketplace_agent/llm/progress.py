from collections.abc import Callable
from time import perf_counter
from typing import Any

from pydantic import BaseModel

from marketplace_agent.llm.base import LLMClient, LLMResponse, Message


class ProgressLLMClient:
    """Report the duration of uncached LLM calls to the terminal."""

    def __init__(
        self,
        inner_client: LLMClient,
        output: Callable[[str], None] = print,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self._inner_client = inner_client
        self._output = output
        self._clock = clock

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        response_schema: type[BaseModel] | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Call the LLM and print how long its response took."""

        model = getattr(self._inner_client, "model", "LLM")
        self._output(f"LLM: запрос к {model}...")
        started_at = self._clock()
        response = self._inner_client.chat(
            messages=messages,
            tools=tools,
            response_schema=response_schema,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        elapsed_seconds = self._clock() - started_at
        self._output(f"LLM: готово за {elapsed_seconds:.1f} с.")
        return response

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Delegate embeddings without progress messages."""

        return self._inner_client.embed(texts)
