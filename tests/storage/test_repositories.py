from pathlib import Path

from marketplace_agent.data_generation.catalog import generate_clean_products
from marketplace_agent.data_generation.reviews import generate_reviews
from marketplace_agent.data_generation.support import generate_orders
from marketplace_agent.storage.database import initialize_database
from marketplace_agent.storage.repositories import (
    OrderRepository,
    ProductRepository,
    ReviewRepository,
)


def test_product_repository_saves_and_loads_product(tmp_path: Path) -> None:
    database_path = tmp_path / "marketplace.db"
    initialize_database(database_path)
    product = generate_clean_products(count=1, seed=7)[0]
    repository = ProductRepository(database_path)

    repository.save_many([product])
    loaded_product = repository.get_by_sku(product.sku)

    assert loaded_product == product

def test_review_repository_saves_and_loads_reviews(tmp_path: Path) -> None:
    database_path = tmp_path / "marketplace.db"
    initialize_database(database_path)
    products = generate_clean_products(count=2, seed=7)
    ProductRepository(database_path).save_many(products)
    reviews = generate_reviews(products, count=3, seed=42)
    repository = ReviewRepository(database_path)

    repository.save_many(reviews)
    loaded_reviews = repository.get_by_sku(reviews[0].sku)

    assert reviews[0] in loaded_reviews


def test_order_repository_loads_orders_for_session(tmp_path: Path) -> None:
    database_path = tmp_path / "marketplace.db"
    initialize_database(database_path)
    products = generate_clean_products(count=2, seed=7)
    ProductRepository(database_path).save_many(products)
    orders = generate_orders(products, count=3, seed=42)
    repository = OrderRepository(database_path)

    repository.save_many(orders)
    loaded_orders = repository.get_by_session(orders[0].session_id)

    assert orders[0] in loaded_orders