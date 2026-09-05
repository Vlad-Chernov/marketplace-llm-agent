from marketplace_agent.data_generation.catalog import (
    generate_clean_products,
    generate_scaled_catalog,
)


def test_generates_same_products_for_same_seed() -> None:
    first_run = generate_clean_products(count=3, seed=42)
    second_run = generate_clean_products(count=3, seed=42)

    assert first_run == second_run


def test_generates_valid_laptop_products() -> None:
    products = generate_clean_products(count=5, seed=7)

    assert len(products) == 5
    assert {product.sku for product in products} == {
        "LAP-0001",
        "LAP-0002",
        "LAP-0003",
        "LAP-0004",
        "LAP-0005",
    }

    for product in products:
        assert product.category == "laptops"
        assert product.price > 0
        assert product.sales_count >= 0
        assert len(product.true_attributes) == 12
        assert product.attributes == product.true_attributes


def test_generates_scaled_catalog_with_three_fixed_categories() -> None:
    products = generate_scaled_catalog(seed=42)

    assert len(products) == 400
    assert sum(product.category == "laptops" for product in products) == 134
    assert sum(product.category == "sneakers" for product in products) == 133
    assert sum(product.category == "kettles" for product in products) == 133
    assert len({product.sku for product in products}) == 400
    assert all(product.sku.startswith("LAP-") for product in products[:134])
    assert all(product.sku.startswith("SNK-") for product in products[134:267])
    assert all(product.sku.startswith("KTL-") for product in products[267:])
