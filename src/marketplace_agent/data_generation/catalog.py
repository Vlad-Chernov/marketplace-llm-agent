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


def generate_scaled_catalog(seed: int) -> list[Product]:
    """Generate the fixed three-category synthetic catalog."""

    rng = Random(seed)
    return [
        *generate_clean_products(count=134, seed=seed),
        *_generate_sneakers(count=133, rng=rng),
        *_generate_kettles(count=133, rng=rng),
    ]


def _generate_sneakers(count: int, rng: Random) -> list[Product]:
    brands_and_series = [
        ("Adidas", "Runflex"),
        ("Nike", "Air Motion"),
        ("Puma", "RS-X"),
        ("New Balance", "Fresh Foam"),
    ]
    products: list[Product] = []

    for index in range(1, count + 1):
        brand, series = rng.choice(brands_and_series)
        true_attributes: dict[str, Any] = {
            "size_eu": rng.choice(["36", "37", "38", "39", "40", "41", "42", "43", "44"]),
            "upper_material": rng.choice(
                ["mesh", "textile", "leather", "synthetic_leather", "suede"]
            ),
            "sole_material": rng.choice(
                ["rubber", "eva", "foam", "polyurethane"]
            ),
            "closure_type": rng.choice(["laces", "velcro", "slip_on"]),
            "season": rng.choice(["summer", "demi_season", "winter"]),
            "color": rng.choice(["black", "white", "gray", "blue", "red", "beige"]),
            "gender": rng.choice(["men", "women", "unisex"]),
            "purpose": rng.choice(
                ["running", "casual", "training", "basketball", "hiking"]
            ),
            "waterproof": rng.choice([True, False]),
            "weight_g": rng.choice([260, 300, 340, 380, 420]),
            "heel_height_mm": rng.choice([20.0, 25.0, 30.0, 35.0]),
            "lining_material": rng.choice(["textile", "mesh", "fleece", "leather"]),
        }
        products.append(
            Product(
                sku=f"SNK-{index:04d}",
                category="sneakers",
                brand=brand,
                model=f"{series} {rng.randint(1000, 9999)}",
                price=Decimal(rng.randrange(4_000, 20_001)),
                sales_count=rng.randint(0, 500),
                supplier_description="",
                attributes=true_attributes.copy(),
                true_attributes=true_attributes,
            )
        )

    return products


def _generate_kettles(count: int, rng: Random) -> list[Product]:
    brands_and_series = [
        ("Tefal", "Element"),
        ("Philips", "Daily"),
        ("Bosch", "Comfort"),
        ("Xiaomi", "Mi Kettle"),
    ]
    products: list[Product] = []

    for index in range(1, count + 1):
        brand, series = rng.choice(brands_and_series)
        true_attributes: dict[str, Any] = {
            "capacity_l": rng.choice([1.0, 1.5, 1.7, 2.0]),
            "power_w": rng.choice([1500, 1800, 2000, 2200]),
            "body_material": rng.choice(
                ["plastic", "glass", "stainless_steel", "ceramic"]
            ),
            "heating_element": rng.choice(["concealed", "open_coil"]),
            "temperature_control": rng.choice([True, False]),
            "keep_warm": rng.choice([True, False]),
            "filter_type": rng.choice(
                ["scale_filter", "removable_filter", "none"]
            ),
            "color": rng.choice(["black", "white", "gray", "silver", "red", "blue"]),
            "cordless": True,
            "auto_shutdown": True,
            "water_level_indicator": rng.choice([True, False]),
            "weight_kg": rng.choice([0.8, 0.9, 1.0, 1.2, 1.4]),
        }
        products.append(
            Product(
                sku=f"KTL-{index:04d}",
                category="kettles",
                brand=brand,
                model=f"{series} {rng.randint(1000, 9999)}",
                price=Decimal(rng.randrange(1_500, 8_001)),
                sales_count=rng.randint(0, 500),
                supplier_description="",
                attributes=true_attributes.copy(),
                true_attributes=true_attributes,
            )
        )

    return products
