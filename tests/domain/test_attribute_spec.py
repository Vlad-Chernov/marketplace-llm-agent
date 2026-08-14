import pytest
from pydantic import ValidationError

from marketplace_agent.domain.models import AttributeSpec


def test_creates_attribute_spec_with_enum_values() -> None:
    spec = AttributeSpec(
        key="ram_gb",
        label="RAM",
        type="enum",
        required=True,
        enum_values=["8", "16", "32"],
        unit="GB",
        synonyms=["оперативная память"],
        affects_filter=True,
    )

    assert spec.key == "ram_gb"
    assert spec.type == "enum"
    assert spec.enum_values == ["8", "16", "32"]


def test_rejects_unknown_attribute_type() -> None:
    with pytest.raises(ValidationError):
        AttributeSpec(
            key="ram_gb",
            label="RAM",
            type="invalid",
            required=True,
        )

def test_rejects_enum_attribute_without_enum_values() -> None:
    with pytest.raises(ValidationError):
        AttributeSpec(
            key="ram_gb",
            label="RAM",
            type="enum",
            required=True,
        )