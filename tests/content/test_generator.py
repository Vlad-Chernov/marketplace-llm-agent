from typing import Any

from pydantic import BaseModel

from marketplace_agent.catalog.extractor import (
    AttributeExtractionResult,
    ExtractedAttribute,
)
from marketplace_agent.content.examples import ContentExample
from marketplace_agent.content.generator import generate_content
from marketplace_agent.data_generation.catalog import generate_clean_products
from marketplace_agent.domain.models import GeneratedContent
from marketplace_agent.llm.base import (
    FakeLLMClient,
    LLMResponse,
    Message,
)


class RecordingFakeLLMClient(FakeLLMClient):
    """Fake client that exposes the last sent prompt."""

    def __init__(
        self,
        chat_responses: list[LLMResponse],
    ) -> None:
        super().__init__(chat_responses=chat_responses)
        self.last_messages: list[Message] = []

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        response_schema: type[BaseModel] | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        self.last_messages = messages
        return super().chat(
            messages=messages,
            tools=tools,
            response_schema=response_schema,
            temperature=temperature,
            max_tokens=max_tokens,
        )


def response() -> LLMResponse:
    return LLMResponse(
        content=(
            '{"title":"Lenovo IdeaPad с 16 GB RAM",'
            '"bullets":["16 GB RAM","512 GB SSD"],'
            '"description":"Ноутбук для работы.",'
            '"keywords":["ноутбук","Lenovo"],'
            '"used_attributes":{"ram_gb":"16","storage_gb":"512"}}'
        ),
        model="fake-model",
        prompt_tokens=1,
        completion_tokens=1,
    )


def test_generates_content_from_confirmed_attributes() -> None:
    product = generate_clean_products(count=1, seed=7)[0]
    extracted_attributes = AttributeExtractionResult(
        attributes={
            "ram_gb": ExtractedAttribute(value="16", confidence=0.95),
            "storage_gb": ExtractedAttribute(
                value="512",
                confidence=0.95,
            ),
        },
        unsupported_facts=[],
    )
    client = FakeLLMClient(chat_responses=[response()])

    content = generate_content(product, extracted_attributes, client)

    assert content.title == "Lenovo IdeaPad с 16 GB RAM"
    assert set(content.used_attributes) <= set(
        extracted_attributes.attributes
    )


def test_includes_examples_as_style_only_context() -> None:
    product = generate_clean_products(count=1, seed=7)[0]
    extracted_attributes = AttributeExtractionResult(
        attributes={
            "ram_gb": ExtractedAttribute(value="16", confidence=0.95),
            "storage_gb": ExtractedAttribute(
                value="512",
                confidence=0.95,
            ),
        },
        unsupported_facts=[],
    )
    examples = [
        ContentExample(
            example_id="EXAMPLE-001",
            category="laptop",
            confirmed_attributes={
                "ram_gb": "16",
                "storage_gb": "512",
            },
            content=GeneratedContent(
                title="Ноутбук для работы",
                bullets=["16 GB RAM"],
                description="Пример карточки.",
                keywords=["ноутбук"],
                used_attributes={
                    "ram_gb": "16",
                    "storage_gb": "512",
                },
            ),
        ),
        ContentExample(
            example_id="EXAMPLE-002",
            category="laptop",
            confirmed_attributes={
                "ram_gb": "16",
                "storage_gb": "256",
            },
            content=GeneratedContent(
                title="Ноутбук для учёбы",
                bullets=["16 GB RAM"],
                description="Ещё один пример.",
                keywords=["ноутбук"],
                used_attributes={
                    "ram_gb": "16",
                    "storage_gb": "256",
                },
            ),
        ),
    ]
    client = RecordingFakeLLMClient(chat_responses=[response()])

    generate_content(
        product,
        extracted_attributes,
        client,
        examples=examples,
    )

    prompt = client.last_messages[0].content
    assert "Примеры задают только стиль" in prompt
    assert "EXAMPLE-001" in prompt
    assert "EXAMPLE-002" in prompt