from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

MessageRole = Literal["system", "user", "assistant", "tool"]


class Message(BaseModel):
    """Represents one message sent to or received from an LLM."""

    model_config = ConfigDict(extra="forbid")

    role: MessageRole
    content: str = Field(min_length=1)


class LLMResponse(BaseModel):
    """Represents a provider-independent LLM response."""

    model_config = ConfigDict(extra="forbid")

    content: str
    model: str
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)


class LLMClient(Protocol):
    """Defines the interface implemented by every LLM provider."""

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        response_schema: type[BaseModel] | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Send chat messages and return a normalized response."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Create one embedding vector for every input text."""


class FakeLLMClient:
    """Return configured responses without any network requests."""

    def __init__(
        self,
        chat_responses: list[LLMResponse] | None = None,
        embeddings: list[list[float]] | None = None,
    ) -> None:
        self._chat_responses = list(chat_responses or [])
        self._embeddings = list(embeddings or [])

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        response_schema: type[BaseModel] | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Return the next configured fake response."""

        del messages, tools, response_schema, temperature, max_tokens

        if not self._chat_responses:
            raise RuntimeError("No fake chat responses are configured.")

        return self._chat_responses.pop(0)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return configured fake embedding vectors."""

        if len(texts) != len(self._embeddings):
            raise ValueError("Embedding count must match text count.")

        return self._embeddings.copy()