from pathlib import Path

from marketplace_agent.evals.content_pipeline import (
    load_content_pipeline_products,
)

MANIFEST_PATH = Path("data/gold/content_pipeline_manifest.json")


def test_loads_reproducible_content_pipeline_products() -> None:
    products = load_content_pipeline_products(MANIFEST_PATH)

    assert [product.sku for product in products] == [
        "LAP-0001",
        "LAP-0002",
        "LAP-0003",
    ]
    assert all(product.supplier_description for product in products)
    assert all(product.true_attributes for product in products)