from datetime import date, timedelta
from random import Random

from marketplace_agent.domain.models import Order, Product


def generate_orders(
    products: list[Product],
    count: int,
    seed: int,
) -> list[Order]:
    """Generate reproducible orders linked to products and customer sessions."""

    if not products:
        raise ValueError("At least one product is required.")
    if count < 0:
        raise ValueError("Order count cannot be negative.")

    rng = Random(seed)
    orders: list[Order] = []
    start_date = date(2025, 1, 1)

    for index in range(count):
        product = rng.choice(products)
        status = rng.choices(
            population=["created", "delivered", "cancelled", "returned"],
            weights=[5, 75, 10, 10],
        )[0]
        purchased_at = start_date + timedelta(days=rng.randint(0, 364))
        delivered_at = (
            purchased_at + timedelta(days=rng.randint(2, 7))
            if status in {"delivered", "returned"}
            else None
        )

        orders.append(
            Order(
                order_id=f"ORD-{index + 1:06d}",
                session_id=f"session-{rng.randint(1, 10):03d}",
                sku=product.sku,
                status=status,
                price=product.price,
                purchased_at=purchased_at,
                delivered_at=delivered_at,
            )
        )

    return orders