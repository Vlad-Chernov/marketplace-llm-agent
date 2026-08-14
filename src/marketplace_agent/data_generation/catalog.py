from decimal import Decimal
from random import Random
from typing import Any

from marketplace_agent.domain.models import Product


def generate_clean_products(count: int, seed: int) -> list[Product]:
    """Generate a reproducible catalog of complete laptop products."""

    if count < 0:
        raise ValueError("Product count cannot be negative.")

    rng = Random(seed)
    products: list[Product] = []

    brands_and_series = [
        ("Acer", "Swift"),
        ("ASUS", "Zenbook"),
        ("Lenovo", "IdeaPad"),
        ("HP", "Pavilion"),
    ]

    for index in range(1, count + 1):
        brand, series = rng.choice(brands_and_series)
        true_attributes: dict[str, Any] = {
            "screen_diagonal_in": rng.choice([13.3, 14.0, 15.6, 16.0]),
            "screen_resolution": rng.choice(
                ["1920x1080", "2560x1600", "2880x1800"]
            ),
            "processor": rng.choice(
                [
                    "Intel Core i5-1340P",
                    "Intel Core i7-1360P",
                    "AMD Ryzen 5 7530U",
                    "AMD Ryzen 7 7730U",
                ]
            ),
            "ram_gb": rng.choice(["8", "16", "32", "64"]),
            "storage_gb": rng.choice(["256", "512", "1024", "2048"]),
            "graphics_card": rng.choice(
                ["Intel Iris Xe", "AMD Radeon Graphics", "NVIDIA RTX 4050"]
            ),
            "operating_system": rng.choice(["Windows 11", "Linux", "No OS"]),
            "weight_kg": rng.choice([1.2, 1.4, 1.6, 1.8, 2.0]),
            "battery_capacity_wh": rng.choice([45, 50, 60, 75]),
            "color": rng.choice(["black", "silver", "gray", "blue"]),
            "keyboard_layout": rng.choice(["RU", "EN", "RU/EN"]),
            "refresh_rate_hz": rng.choice(["60", "90", "120", "144"]),
        }

        products.append(
            Product(
                sku=f"LAP-{index:04d}",
                category="laptops",
                brand=brand,
                model=f"{series} {rng.randint(1000, 9999)}",
                price=Decimal(rng.randrange(50_000, 200_001)),
                sales_count=rng.randint(0, 500),
                supplier_description="",
                attributes=true_attributes.copy(),
                true_attributes=true_attributes,
            )
        )

    return products