from marketplace_agent.data_generation.catalog import (
    generate_clean_products,
    generate_scaled_catalog,
)
from marketplace_agent.data_generation.reviews import generate_reviews


def test_generates_same_reviews_for_same_seed() -> None:
    products = generate_clean_products(count=5, seed=7)

    first_run = generate_reviews(products, count=20, seed=42)
    second_run = generate_reviews(products, count=20, seed=42)

    assert first_run == second_run


def test_generates_reviews_linked_to_catalog_products() -> None:
    products = generate_clean_products(count=5, seed=7)

    reviews = generate_reviews(products, count=20, seed=42)

    product_skus = {product.sku for product in products}

    assert len(reviews) == 20
    assert {review.sku for review in reviews} <= product_skus
    assert all(1 <= review.rating <= 5 for review in reviews)


def test_generates_expected_review_profiles() -> None:
    products = generate_clean_products(count=40, seed=7)

    reviews = generate_reviews(products, count=200, seed=42)

    defect_ratio = sum(review.defect_label is not None for review in reviews) / 200
    personal_data_ratio = (
        sum(review.contains_personal_data for review in reviews) / 200
    )
    delivery_ratio = sum(review.is_delivery_review for review in reviews) / 200

    assert 0.10 <= defect_ratio <= 0.20
    assert 0.02 <= personal_data_ratio <= 0.08
    assert 0.01 <= delivery_ratio <= 0.06

def test_uses_category_name_in_scaled_catalog_reviews() -> None:
    products = generate_scaled_catalog(seed=7)
    reviews = generate_reviews(products, count=4_000, seed=42)
    products_by_sku = {product.sku: product for product in products}

    for review in reviews:
        category = products_by_sku[review.sku].category
        if category == "sneakers":
            assert "Ноутбук" not in review.text
        if category == "kettles":
            assert "Ноутбук" not in review.text