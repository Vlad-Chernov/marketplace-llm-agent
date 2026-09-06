import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, model_validator

from marketplace_agent.domain.models import GeneratedContent


class ContentExample(BaseModel):
    """One verified card used as a few-shot style example."""

    model_config = ConfigDict(extra="forbid")

    example_id: str
    category: str
    confirmed_attributes: dict[str, str | int | float | bool]
    content: GeneratedContent

    @model_validator(mode="after")
    def validate_used_attributes(self) -> "ContentExample":
        for key, value in self.content.used_attributes.items():
            if self.confirmed_attributes.get(key) != value:
                raise ValueError(
                    "used_attributes must contain only confirmed values."
                )
        return self


def load_content_examples(
    path: Path | str,
) -> list[ContentExample]:
    """Load static few-shot examples from JSON."""

    records = json.loads(Path(path).read_text(encoding="utf-8"))
    examples = [ContentExample.model_validate(record) for record in records]
    example_ids = [example.example_id for example in examples]

    if len(set(example_ids)) != len(example_ids):
        raise ValueError("Duplicate content example ID.")

    return examples


def select_similar_examples(
    category: str,
    confirmed_attributes: Mapping[str, object],
    examples: Sequence[ContentExample],
    limit: int = 2,
) -> list[ContentExample]:
    """Return examples ranked by category and matching attributes."""

    if limit < 1:
        raise ValueError("limit must be at least 1.")

    def score(example: ContentExample) -> int:
        category_score = 100 if example.category == category else 0
        attribute_score = sum(
            example.confirmed_attributes.get(key) == value
            for key, value in confirmed_attributes.items()
        )
        return category_score + attribute_score

    ranked_examples = sorted(
        examples,
        key=lambda example: (-score(example), example.example_id),
    )
    return ranked_examples[:limit]