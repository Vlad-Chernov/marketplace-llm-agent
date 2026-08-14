from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from marketplace_agent.domain.models import Order


def test_order_serializes_to_json() -> None:
    order = Order(
        order_id="ORD-000001",
        session_id="session-001",
        sku="LAP-0001",
        status="delivered",
        price=Decimal("99999.90"),
        purchased_at=date(2026, 1, 5),
        delivered_at=date(2026, 1, 8),
    )

    serialized = order.model_dump(mode="json")

    assert serialized["order_id"] == "ORD-000001"
    assert serialized["price"] == "99999.90"
    assert serialized["delivered_at"] == "2026-01-08"


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("order_id", "order-1"),
        ("price", Decimal(0)),
        ("status", "unknown"),
    ],
)
def test_order_rejects_invalid_core_fields(
    field_name: str,
    value: object,
) -> None:
    data: dict[str, object] = {
        "order_id": "ORD-000001",
        "session_id": "session-001",
        "sku": "LAP-0001",
        "status": "delivered",
        "price": Decimal("99999.90"),
        "purchased_at": date(2026, 1, 5),
        "delivered_at": date(2026, 1, 8),
    }
    data[field_name] = value

    with pytest.raises(ValidationError):
        Order(**data)