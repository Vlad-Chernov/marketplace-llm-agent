import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from marketplace_agent.domain.models import Review
from marketplace_agent.llm.base import LLMClient, Message
from marketplace_agent.llm.structured import chat_structured
from marketplace_agent.reviews.taxonomy import DefectTaxonomy

ReviewKind = Literal[
    "product_defect",
    "delivery_or_service",
    "no_defect",
]
Severity = Literal["low", "medium", "high"]


class ReviewLabels(BaseModel):
    """Store validated labels for one review."""

    model_config = ConfigDict(extra="forbid")

    defect_id: str = Field(min_length=1)
    kind: ReviewKind
    severity: Severity | None
    confidence: float = Field(ge=0.0, le=1.0)


class ReviewClassificationError(ValueError):
    """Raised when labels contradict the fixed taxonomy."""


def classify_review(
    review: Review,
    taxonomy: DefectTaxonomy,
    llm: LLMClient,
) -> ReviewLabels:
    """Classify one review using a fixed defect taxonomy."""

    prompt_path = Path(__file__).parent / "prompts" / "classify_review.md"
    prompt = prompt_path.read_text(encoding="utf-8").format(
        review=json.dumps(
            {
                "rating": review.rating,
                "text": review.text,
            },
            ensure_ascii=False,
        ),
        taxonomy=json.dumps(
            [category.model_dump() for category in taxonomy.categories],
            ensure_ascii=False,
        ),
    )
    labels = chat_structured(
        client=llm,
        messages=[Message(role="user", content=prompt)],
        response_schema=ReviewLabels,
        max_retries=1,
    )
    _validate_labels(labels, taxonomy)
    return labels


def _validate_labels(
    labels: ReviewLabels,
    taxonomy: DefectTaxonomy,
) -> None:
    category_ids = {category.id for category in taxonomy.categories}
    if labels.defect_id not in category_ids:
        raise ReviewClassificationError(
            f"Unknown defect ID: {labels.defect_id}"
        )

    if labels.kind == "product_defect":
        if labels.defect_id == "no_defect" or labels.severity is None:
            raise ReviewClassificationError(
                "Product defect requires a defect ID and severity."
            )
        return

    if labels.defect_id != "no_defect" or labels.severity is not None:
        raise ReviewClassificationError(
            "Non-product review requires no_defect and no severity."
        )