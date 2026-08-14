import argparse
import hashlib
import json
import sqlite3
from pathlib import Path
from random import Random
from typing import Any

from marketplace_agent.data_generation.catalog import generate_clean_products
from marketplace_agent.data_generation.noise import noise_product
from marketplace_agent.data_generation.reviews import generate_reviews
from marketplace_agent.data_generation.support import generate_orders
from marketplace_agent.storage.database import initialize_database
from marketplace_agent.storage.repositories import (
    OrderRepository,
    ProductRepository,
    ReviewRepository,
)


def generate_dataset(
    database_path: Path,
    manifest_path: Path,
    seed: int,
) -> None:
    """Generate and persist the complete MVP synthetic dataset."""

    clean_products = generate_clean_products(count=40, seed=seed)
    rng = Random(seed)
    products = [noise_product(product, rng) for product in clean_products]
    reviews = generate_reviews(products, count=200, seed=seed)
    orders = generate_orders(products, count=60, seed=seed)

    initialize_database(database_path)
    ProductRepository(database_path).save_many(products)
    ReviewRepository(database_path).save_many(reviews)
    OrderRepository(database_path).save_many(orders)

    manifest = {
        "generator_version": "1.0",
        "seed": seed,
        "counts": {
            "products": len(products),
            "reviews": len(reviews),
            "orders": len(orders),
        },
        "files": {
            "database_sha256": calculate_sha256(database_path),
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

def build_dataset_profile(database_path: Path) -> dict[str, Any]:
    """Return summary counts and rating distribution from SQLite."""

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
        average_rating = connection.execute(
            "SELECT AVG(rating) FROM reviews"
        ).fetchone()[0]
        rating_rows = connection.execute(
            "SELECT rating, COUNT(*) FROM reviews GROUP BY rating ORDER BY rating"
        ).fetchall()

    return {
        "products": product_count,
        "reviews": review_count,
        "orders": order_count,
        "average_rating": round(float(average_rating), 2),
        "ratings": {str(rating): count for rating, count in rating_rows},
    }

def calculate_sha256(path: Path) -> str:
    """Return the SHA-256 checksum of one file."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main() -> None:
    """Generate the default local MVP dataset."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("data/synthetic/marketplace.db"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/MANIFEST.json"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_dataset(
        database_path=args.database,
        manifest_path=args.manifest,
        seed=args.seed,
    )

    profile = build_dataset_profile(args.database)
    print(
        "Profile: "
        f"{profile['products']} products, "
        f"{profile['reviews']} reviews, "
        f"{profile['orders']} orders."
    )
    
    print(f"Dataset written to {args.database}")


if __name__ == "__main__":
    main()