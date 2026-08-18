from pathlib import Path
from random import Random

from marketplace_agent.catalog.extractor import extract_attributes
from marketplace_agent.config import Settings
from marketplace_agent.content.generator import generate_content
from marketplace_agent.data_generation.catalog import generate_clean_products
from marketplace_agent.data_generation.noise import noise_product
from marketplace_agent.data_generation.specs import load_attribute_specs
from marketplace_agent.llm.factory import create_llm_client


def main() -> None:
    """Run one real baseline content-generation pipeline."""

    product = noise_product(
        generate_clean_products(count=1, seed=7)[0],
        Random(42),
    )
    client = create_llm_client(Settings.from_environment())
    specs = load_attribute_specs(Path("data/specs/laptops.yaml"))
    extracted = extract_attributes(product, specs, client)
    content = generate_content(product, extracted, client)

    print("Generated content:")
    print(content.model_dump_json(indent=2))


if __name__ == "__main__":
    main()