import json
from pathlib import Path

from pydantic import BaseModel, Field

from marketplace_agent.domain.models import AttributeSpec, Product
from marketplace_agent.llm.base import LLMClient, Message
from marketplace_agent.llm.structured import chat_structured


class ExtractedAttribute(BaseModel):
    """Store one extracted value and model confidence."""

    value: str | int | float | bool | None
    confidence: float = Field(ge=0.0, le=1.0)


class AttributeExtractionResult(BaseModel):
    """Store all extracted attributes and unsupported facts."""

    attributes: dict[str, ExtractedAttribute]
    unsupported_facts: list[str]


def extract_attributes(
    product: Product,
    attribute_specs: list[AttributeSpec],
    client: LLMClient,
) -> AttributeExtractionResult:
    """Extract category attributes from one supplier description."""

    prompt_path = Path(__file__).parent / "prompts" / "extract_attributes.md"
    prompt = prompt_path.read_text(encoding="utf-8").format(
        description=product.supplier_description,
        attribute_specs=json.dumps(
            [spec.model_dump() for spec in attribute_specs],
            ensure_ascii=False,
        ),
    )

    return chat_structured(
        client=client,
        messages=[Message(role="user", content=prompt)],
        response_schema=AttributeExtractionResult,
        max_retries=1,
    )