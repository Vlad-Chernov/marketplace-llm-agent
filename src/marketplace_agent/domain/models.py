from decimal import Decimal
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

AttributeType = Literal["string", "integer", "number", "boolean", "enum"]


class AttributeSpec(BaseModel):
    """Describes one product attribute allowed in a category."""

    model_config = ConfigDict(extra="forbid")

    key: str
    label: str
    type: AttributeType
    required: bool
    enum_values: list[str] = Field(default_factory=list)
    unit: str | None = None
    synonyms: list[str] = Field(default_factory=list)
    affects_filter: bool = False

    @model_validator(mode="after")
    def validate_enum_values(self) -> Self:
        if self.type == "enum" and not self.enum_values:
            raise ValueError("Enum attributes require at least one enum value.")
        return self

class Product(BaseModel):
    """Represents one marketplace product."""

    model_config = ConfigDict(extra="forbid")

    sku: str = Field(pattern=r"^[A-Z0-9][A-Z0-9-]*$", min_length=3)
    category: str = Field(min_length=1)
    brand: str = Field(min_length=1)
    model: str = Field(min_length=1)
    price: Decimal = Field(gt=0)
    sales_count: int = Field(ge=0)
    supplier_description: str
    attributes: dict[str, Any] = Field(default_factory=dict)
    true_attributes: dict[str, Any] = Field(
        default_factory=dict,
        exclude=True,
        repr=False,
    )


class GeneratedContent(BaseModel):
    """Contains marketplace content generated from confirmed attributes."""

    model_config = ConfigDict(extra="forbid")

    title: str
    bullets: list[str]
    description: str
    keywords: list[str]
    used_attributes: dict[str, Any]


ViolationSeverity = Literal["low", "medium", "high", "critical"]


class RuleViolation(BaseModel):
    """Describes one content-rule violation."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    message: str
    severity: ViolationSeverity
    field: str | None = None


PipelineStatus = Literal["completed", "manual_review", "failed"]


class PipelineResult(BaseModel):
    """Contains the outcome of the content-processing pipeline."""

    model_config = ConfigDict(extra="forbid")

    sku: str
    content: GeneratedContent | None = None
    violations: list[RuleViolation] = Field(default_factory=list)
    attempts: int = Field(ge=0)
    status: PipelineStatus