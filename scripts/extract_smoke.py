from pathlib import Path
from random import Random

from marketplace_agent.catalog.extractor import extract_attributes
from marketplace_agent.config import Settings
from marketplace_agent.data_generation.catalog import generate_clean_products
from marketplace_agent.data_generation.noise import noise_product
from marketplace_agent.data_generation.specs import load_attribute_specs
from marketplace_agent.llm.factory import create_llm_client


def main() -> None:
    """Run one real attribute-extraction request."""

    product = noise_product(
        generate_clean_products(count=1, seed=7)[0],
        Random(42),
    )
    specs = load_attribute_specs(Path("data/specs/laptops.yaml"))
    client = create_llm_client(Settings.from_environment())

    result = extract_attributes(product, specs, client)

    print("Supplier description:")
    print(product.supplier_description)
    print("\nExtracted attributes:")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()