from marketplace_agent.data_generation.catalog import generate_clean_products


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