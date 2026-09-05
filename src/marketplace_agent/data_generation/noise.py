from random import Random
from typing import Any

from marketplace_agent.domain.models import Product

ATTRIBUTE_LABELS = {
    "screen_diagonal_in": "Диагональ экрана",
    "screen_resolution": "Разрешение экрана",
    "processor": "Процессор",
    "ram_gb": "Оперативная память",
    "storage_gb": "Накопитель",
    "graphics_card": "Видеокарта",
    "operating_system": "Операционная система",
    "weight_kg": "Вес",
    "battery_capacity_wh": "Ёмкость аккумулятора",
    "color": "Цвет",
    "keyboard_layout": "Раскладка клавиатуры",
    "refresh_rate_hz": "Частота обновления",
}

CATEGORY_NAMES = {
    "laptops": "Ноутбук",
    "sneakers": "Кроссовки",
    "kettles": "Электрочайник",
}

def noise_product(product: Product, rng: Random) -> Product:
    """Create a noisy supplier version without changing hidden true attributes."""

    noisy_attributes: dict[str, Any] = {
        key: value
        for key, value in product.attributes.items()
        if rng.random() >= 0.40
    }

    facts = [
        f"{ATTRIBUTE_LABELS.get(key, key.replace('_', ' '))}: {value}"
        for key, value in product.true_attributes.items()
    ]
    product_name = CATEGORY_NAMES[product.category]
    description = (
        f"{product_name} {product.brand} {product.model}. "
        + ". ".join(facts)
        + "."
    )

    if rng.random() < 0.10:
        description = description.replace("Ноутбук", "Ноутбк")
    if rng.random() < 0.10:
        description = description.upper()
    if rng.random() < 0.10:
        description = f"<p>{description}</p>"

    return product.model_copy(
        update={
            "attributes": noisy_attributes,
            "supplier_description": description,
        }
    )