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

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert product_count == 40
    assert review_count == 200
    assert order_count == 60
    assert manifest["seed"] == 42
    assert manifest["counts"] == {
        "products": 40,
        "reviews": 200,
        "orders": 60,
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

    assert profile["products"] == 40
    assert profile["reviews"] == 200
    assert profile["orders"] == 60
    assert sum(profile["ratings"].values()) == 200