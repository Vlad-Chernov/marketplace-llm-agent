import json
import sqlite3
from pathlib import Path

from marketplace_agent.data_generation.generate import (
    build_dataset_profile,
    generate_dataset,
)


def test_generates_database_and_manifest(tmp_path: Path) -> None:
    database_path = tmp_path / "marketplace.db"
    manifest_path = tmp_path / "MANIFEST.json"

    generate_dataset(
        database_path=database_path,
        manifest_path=manifest_path,
        seed=42,
    )

    with sqlite3.connect(database_path) as connection:
        product_count = connection.execute(
            "SELECT COUNT(*) FROM products"
        ).fetchone()[0]
        review_count = connection.execute(
            "SELECT COUNT(*) FROM reviews"
        ).fetchone()[0]
        order_count = connection.execute(
            "SELECT COUNT(*) FROM orders"
        ).fetchone()[0]
        unlinked_review_count = connection.execute(
            "SELECT COUNT(*) "
            "FROM reviews r LEFT JOIN products p ON p.sku = r.sku "
            "WHERE p.sku IS NULL"
        ).fetchone()[0]
        unlinked_order_count = connection.execute(
            "SELECT COUNT(*) "
            "FROM orders o LEFT JOIN products p ON p.sku = o.sku "
            "WHERE p.sku IS NULL"
        ).fetchone()[0]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert product_count == 400
    assert review_count == 4_000
    assert order_count == 500
    assert unlinked_review_count == 0
    assert unlinked_order_count == 0
    assert manifest["seed"] == 42
    assert manifest["counts"] == {
        "products": 400,
        "reviews": 4_000,
        "orders": 500,
    }
    assert manifest["files"]["database_sha256"]

def test_builds_dataset_profile(tmp_path: Path) -> None:
    database_path = tmp_path / "marketplace.db"
    manifest_path = tmp_path / "MANIFEST.json"
    generate_dataset(
        database_path=database_path,
        manifest_path=manifest_path,
        seed=42,
    )

    profile = build_dataset_profile(database_path)

    assert profile["products"] == 400
    assert profile["reviews"] == 4_000
    assert profile["orders"] == 500
    assert profile["categories"] == {
        "kettles": 133,
        "laptops": 134,
        "sneakers": 133,
    }
    assert sum(profile["ratings"].values()) == 4_000
    assert profile["defect_reviews"] == 600
    assert profile["reviews_with_personal_data"] == 200
    assert profile["delivery_reviews"] == 104
    assert sum(profile["order_statuses"].values()) == 500
    assert set(profile["order_statuses"]) == {
        "cancelled",
        "created",
        "delivered",
        "returned",
    }
    assert profile["review_shares"] == {
        "defect": 0.15,
        "personal_data": 0.05,
        "delivery": 0.026,
    }