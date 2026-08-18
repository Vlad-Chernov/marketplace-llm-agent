from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

GoldenCaseType = Literal[
    "attribute_extraction",
    "content_generation",
    "validation",
    "review_analysis",
    "support",
    "no_answer",
    "adversarial",
]

GoldenCaseOrigin = Literal["manual", "synthetic"]


class GoldenCase(BaseModel):
    """Represent one fixed quality-evaluation task."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    type: GoldenCaseType
    expected_answer: Any
    expected_citations: list[str] = Field(default_factory=list)
    must_call_tools: list[str] = Field(default_factory=list)
    must_not_contain: list[str] = Field(default_factory=list)
    origin: GoldenCaseOrigin
    notes: str = ""