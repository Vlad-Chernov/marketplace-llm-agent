from decimal import Decimal

import pytest
from pydantic import ValidationError

from marketplace_agent.domain.models import (
    GeneratedContent,
    PipelineAttempt,
    PipelineResult,
    Product,
    RuleViolation,
)


def create_product(**changes: object) -> Product:
    data: dict[str, object] = {
        "sku": "LAP-0001",
        "category": "laptops",
        "brand": "Acer",
        "model": "Swift 14",
        "price": Decimal("99999.90"),
        "sales_count": 12,
        "supplier_description": "14-inch laptop",
        "attributes": {"ram_gb": "16"},
        "true_attributes": {"ram_gb": "16", "storage_gb": "512"},
    }
    data.update(changes)
    return Product(**data)


def test_product_hides_true_attributes_in_json() -> None:
    product = create_product()

    serialized = product.model_dump(mode="json")

    assert serialized["price"] == "99999.90"
    assert serialized["attributes"] == {"ram_gb": "16"}
    assert "true_attributes" not in serialized


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("sku", "laptop 1"),
        ("category", ""),
        ("price", Decimal(0)),
    ],
)
def test_product_rejects_invalid_core_fields(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises(ValidationError):
        create_product(**{field_name: value})


def test_pipeline_result_serializes_content_and_violations() -> None:
    content = GeneratedContent(
        title="Acer Swift 14",
        bullets=["16 GB RAM", "512 GB SSD"],
        description="Compact laptop for daily work.",
        keywords=["laptop", "acer"],
        used_attributes={"ram_gb": "16"},
    )
    violation = RuleViolation(
        rule_id="title-length",
        message="Title is too long.",
        severity="medium",
        field="title",
    )
    result = PipelineResult(
        sku="LAP-0001",
        content=content,
        violations=[violation],
        attempts=1,
        status="completed",
    )

    assert result.model_dump(mode="json")["content"]["title"] == "Acer Swift 14"
    assert result.violations[0].rule_id == "title-length"


def test_pipeline_result_serializes_attempt_history() -> None:
    result = PipelineResult(
        sku="LAP-0001",
        attempts=2,
        status="manual_review",
        attempt_history=[
            PipelineAttempt(attempt=1, violation_count=2, status="invalid"),
            PipelineAttempt(attempt=2, violation_count=1, status="invalid"),
        ],
    )

    assert result.model_dump(mode="json")["attempt_history"] == [
        {"attempt": 1, "violation_count": 2, "status": "invalid"},
        {"attempt": 2, "violation_count": 1, "status": "invalid"},
    ]
