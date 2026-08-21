from datetime import date
from decimal import Decimal

from marketplace_agent.domain.models import Order
from marketplace_agent.support.eligibility import (
    ReturnRequest,
    check_return_eligibility,
)


def make_order(
    *,
    status: str = "delivered",
    delivered_at: date | None = date(2026, 8, 1),
) -> Order:
    return Order(
        order_id="ORD-000001",
        session_id="session-001",
        sku="LAP-0001",
        status=status,
        price=Decimal(79990),
        purchased_at=date(2026, 7, 30),
        delivered_at=delivered_at,
    )


def test_allows_good_quality_return_at_fourteen_day_boundary() -> None:
    result = check_return_eligibility(
        make_order(),
        ReturnRequest(
            condition="preserved",
            defect_confirmed=False,
        ),
        now=date(2026, 8, 15),
    )

    assert result.status == "allowed"
    assert result.reason == "Срок возврата и состояние товара соответствуют правилам."


def test_denies_late_or_damaged_return() -> None:
    late = check_return_eligibility(
        make_order(),
        ReturnRequest(
            condition="preserved",
            defect_confirmed=False,
        ),
        now=date(2026, 8, 16),
    )
    damaged = check_return_eligibility(
        make_order(),
        ReturnRequest(
            condition="damaged",
            defect_confirmed=False,
        ),
        now=date(2026, 8, 5),
    )

    assert late.status == "denied"
    assert damaged.status == "denied"


def test_requests_clarification_for_unknown_delivery_or_defect() -> None:
    unknown_delivery = check_return_eligibility(
        make_order(delivered_at=None),
        ReturnRequest(
            condition="preserved",
            defect_confirmed=False,
        ),
        now=date(2026, 8, 5),
    )
    defect = check_return_eligibility(
        make_order(),
        ReturnRequest(
            condition="preserved",
            defect_confirmed=True,
        ),
        now=date(2026, 8, 5),
    )

    assert unknown_delivery.status == "needs_clarification"
    assert defect.status == "needs_clarification"