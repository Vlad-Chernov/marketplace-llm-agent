import json
from pathlib import Path

from marketplace_agent.catalog.extractor import AttributeExtractionResult
from marketplace_agent.domain.models import GeneratedContent, Product
from marketplace_agent.llm.base import LLMClient, Message
from marketplace_agent.llm.structured import chat_structured


def generate_content(
    product: Product,
    extracted_attributes: AttributeExtractionResult,
    client: LLMClient,
) -> GeneratedContent:
    """Generate a product card using only confirmed extracted attributes."""

    confirmed_attributes = {
        key: attribute.value
        for key, attribute in extracted_attributes.attributes.items()
        if attribute.value is not None
    }
    prompt_path = Path(__file__).parent / "prompts" / "generate_content.md"
    prompt = prompt_path.read_text(encoding="utf-8").format(
        product=json.dumps(
            {
                "brand": product.brand,
                "model": product.model,
                "category": product.category,
            },
            ensure_ascii=False,
        ),
        attributes=json.dumps(confirmed_attributes, ensure_ascii=False),
    )

    return chat_structured(
        client=client,
        messages=[Message(role="user", content=prompt)],
        response_schema=GeneratedContent,
        max_retries=1,
    )