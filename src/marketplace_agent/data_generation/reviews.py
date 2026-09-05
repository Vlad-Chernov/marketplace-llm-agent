from datetime import date, timedelta
from random import Random

from marketplace_agent.domain.models import Product, Review

DEFECT_TEXTS = {
    "battery_drain": "Батарея слишком быстро разряжается.",
    "overheating": "Ноутбук сильно перегревается во время работы.",
    "screen_flicker": "Экран периодически мерцает.",
    "keyboard_failure": "Несколько клавиш перестали работать.",
    "wifi_disconnect": "Wi-Fi постоянно отключается.",
}

PRODUCT_NAMES = {
    "laptops": "Ноутбук",
    "sneakers": "Кроссовки",
    "kettles": "Электрочайник",
}


def generate_reviews(
    products: list[Product],
    count: int,
    seed: int,
) -> list[Review]:
    """Generate reproducible Russian-language reviews for catalog products."""

    if not products:
        raise ValueError("At least one product is required.")
    if count < 0:
        raise ValueError("Review count cannot be negative.")

    rng = Random(seed)
    reviews: list[Review] = []
    defect_count = round(count * 0.15)
    emerging_skus = {product.sku for product in products[:3]}
    start_date = date(2025, 1, 1)

    for index in range(count):
        product = rng.choice(products)
        product_name = PRODUCT_NAMES[product.category]
        defect_label = (
            list(DEFECT_TEXTS)[index % len(DEFECT_TEXTS)]
            if index < defect_count
            else None
        )
        is_delivery_review = (
            index >= defect_count
            and (index - defect_count) % 33 == 0
        )
        contains_personal_data = index % 20 == 0
        if is_delivery_review:
            text = (
                f"{product_name} ещё не оценил: "
                "курьер задержал доставку."
            )
            rating = 3
        elif defect_label is not None:
            if product.category == "laptops":
                text = DEFECT_TEXTS[defect_label]
            else:
                defect_label = "other"
                text = f"{product_name} имеет производственный дефект."
            rating = rng.choice([1, 2])
        else:
            rating = rng.choices(
                population=[1, 2, 3, 4, 5],
                weights=[20, 7, 10, 18, 55],
            )[0]
            text = (
                f"{product_name} полностью соответствует описанию."
                if rating >= 4
                else "Пользуюсь несколько дней, впечатления смешанные."
            )

        if contains_personal_data:
            text += " Мой номер: +7 999 123-45-67."

        minimum_day = 250 if (
            defect_label == "overheating" and product.sku in emerging_skus
        ) else 0
        created_at = start_date + timedelta(
            days=rng.randint(minimum_day, 364)
        )

        helpful_count = 0 if index < round(count * 0.40) else rng.randint(1, 20)

        reviews.append(
            Review(
                review_id=f"REV-{index + 1:06d}",
                sku=product.sku,
                rating=rating,
                text=text,
                created_at=created_at,
                helpful_count=helpful_count,
                defect_label=defect_label,
                contains_personal_data=contains_personal_data,
                is_delivery_review=is_delivery_review,
            )
        )

    return reviews