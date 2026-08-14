from typing import Literal, Self

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