from random import Random

from marketplace_agent.data_generation.catalog import generate_clean_products
from marketplace_agent.data_generation.noise import noise_product


def test_noise_preserves_hidden_truth_and_removes_only_known_attributes() -> None:
    clean_product = generate_clean_products(count=1, seed=7)[0]

    noisy_product = noise_product(clean_product, Random(42))

    assert noisy_product.true_attributes == clean_product.true_attributes
    assert set(noisy_product.attributes) <= set(clean_product.true_attributes)
    assert noisy_product.sku == clean_product.sku
    assert noisy_product.price == clean_product.price


def test_noise_is_reproducible_for_same_random_seed() -> None:
    clean_product = generate_clean_products(count=1, seed=7)[0]

    first_result = noise_product(clean_product, Random(42))
    second_result = noise_product(clean_product, Random(42))

    assert first_result == second_result


def test_noise_removes_about_forty_percent_of_attributes() -> None:
    clean_products = generate_clean_products(count=40, seed=7)
    rng = Random(42)
    noisy_products = [noise_product(product, rng) for product in clean_products]

    total_attributes = sum(len(product.true_attributes) for product in clean_products)
    removed_attributes = sum(
        len(product.true_attributes) - len(product.attributes)
        for product in noisy_products
    )

    removed_ratio = removed_attributes / total_attributes

    assert 0.30 <= removed_ratio <= 0.50