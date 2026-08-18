from marketplace_agent.catalog.extractor import (
    AttributeExtractionResult,
    ExtractedAttribute,
)
from marketplace_agent.content.generator import generate_content
from marketplace_agent.data_generation.catalog import generate_clean_products
from marketplace_agent.llm.base import FakeLLMClient, LLMResponse


def test_generates_content_from_confirmed_attributes() -> None:
    product = generate_clean_products(count=1, seed=7)[0]
    extracted_attributes = AttributeExtractionResult(
        attributes={
            "ram_gb": ExtractedAttribute(value="16", confidence=0.95),
            "storage_gb": ExtractedAttribute(value="512", confidence=0.95),
        },
        unsupported_facts=[],
    )
    client = FakeLLMClient(
        chat_responses=[
            LLMResponse(
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
        ]
    )

    content = generate_content(product, extracted_attributes, client)

    assert content.title == "Lenovo IdeaPad с 16 GB RAM"
    assert set(content.used_attributes) <= set(extracted_attributes.attributes)