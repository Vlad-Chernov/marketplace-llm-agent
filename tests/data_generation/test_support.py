from marketplace_agent.data_generation.catalog import generate_clean_products
from marketplace_agent.data_generation.support import generate_orders


def test_generates_same_orders_for_same_seed() -> None:
    products = generate_clean_products(count=5, seed=7)

    first_run = generate_orders(products, count=10, seed=42)
    second_run = generate_orders(products, count=10, seed=42)

    assert first_run == second_run


def test_generates_orders_linked_to_products_and_sessions() -> None:
    products = generate_clean_products(count=5, seed=7)

    orders = generate_orders(products, count=10, seed=42)

    prices_by_sku = {product.sku: product.price for product in products}

    assert len(orders) == 10
    assert len({order.order_id for order in orders}) == 10

    for order in orders:
        assert order.sku in prices_by_sku
        assert order.price == prices_by_sku[order.sku]
        assert order.session_id.startswith("session-")