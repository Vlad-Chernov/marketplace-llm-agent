from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict

from marketplace_agent.domain.models import Order

RETURN_WINDOW_DAYS = 14


class ReturnRequest(BaseModel):
    """Store return details supplied by the customer."""

    model_config = ConfigDict(extra="forbid")

    condition: Literal["preserved", "damaged", "unknown"]
    defect_confirmed: bool | None


class EligibilityResult(BaseModel):
    """Store a deterministic return-eligibility decision."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["allowed", "denied", "needs_clarification"]
    reason: str


def check_return_eligibility(
    order: Order,
    request: ReturnRequest,
    now: date,
) -> EligibilityResult:
    """Decide return eligibility from fixed policy rules."""

    if order.status in {"cancelled", "returned"}:
        return EligibilityResult(
            status="denied",
            reason="Для отменённого или уже возвращённого заказа возврат недоступен.",
        )

    if order.status != "delivered" or order.delivered_at is None:
        return EligibilityResult(
            status="needs_clarification",
            reason="Нужна подтверждённая дата получения заказа.",
        )

    if request.defect_confirmed is None:
        return EligibilityResult(
            status="needs_clarification",
            reason="Нужно подтверждение наличия производственного дефекта.",
        )

    if request.defect_confirmed:
        return EligibilityResult(
            status="needs_clarification",
            reason="Нужна проверка гарантийного срока после диагностики.",
        )

    if request.condition == "unknown":
        return EligibilityResult(
            status="needs_clarification",
            reason="Нужно уточнить состояние и комплектность товара.",
        )

    if request.condition == "damaged":
        return EligibilityResult(
            status="denied",
            reason="Товарный вид или комплектность не сохранены.",
        )

    days_since_delivery = (now - order.delivered_at).days
    if days_since_delivery > RETURN_WINDOW_DAYS:
        return EligibilityResult(
            status="denied",
            reason="Срок возврата 14 календарных дней истёк.",
        )

    return EligibilityResult(
        status="allowed",
        reason="Срок возврата и состояние товара соответствуют правилам.",
    )