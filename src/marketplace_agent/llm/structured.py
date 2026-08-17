from typing import TypeVar

from pydantic import BaseModel, ValidationError

from marketplace_agent.llm.base import LLMClient, Message

StructuredModel = TypeVar("StructuredModel", bound=BaseModel)


class StructuredOutputError(RuntimeError):
    """Raised when an LLM response cannot match the required schema."""


def chat_structured(
    client: LLMClient,
    messages: list[Message],
    response_schema: type[StructuredModel],
    max_retries: int,
) -> StructuredModel:
    """Call an LLM and validate its JSON response with a Pydantic model."""

    last_error: ValidationError | None = None
    current_messages = list(messages)

    for attempt in range(max_retries + 1):
        response = client.chat(
            messages=current_messages,
            tools=None,
            response_schema=response_schema,
            temperature=0.0,
            max_tokens=512,
        )

        try:
            return response_schema.model_validate_json(response.content)
        except ValidationError as error:
            last_error = error

            if attempt < max_retries:
                current_messages.append(
                    Message(
                        role="system",
                        content=(
                            "Предыдущий ответ не прошёл проверку. "
                            "Верни только валидный JSON по указанной схеме."
                        ),
                    )
                )

    raise StructuredOutputError(
        "LLM returned invalid structured output after allowed retries."
    ) from last_error