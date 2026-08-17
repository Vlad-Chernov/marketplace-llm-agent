from pathlib import Path

from marketplace_agent.catalog.extractor import extract_attributes
from marketplace_agent.data_generation.catalog import generate_clean_products
from marketplace_agent.data_generation.specs import load_attribute_specs
from marketplace_agent.llm.base import FakeLLMClient, LLMResponse


def test_extractor_keeps_missing_attribute_empty() -> None:
    product = generate_clean_products(count=1, seed=7)[0].model_copy(
        update={
            "supplier_description": "Ноутбук Acer Swift. Оперативная память 16 GB.",
            "attributes": {},
        }
    )
    specs = load_attribute_specs(Path("data/specs/laptops.yaml"))
    client = FakeLLMClient(
        chat_responses=[
            LLMResponse(
                content=(
                    '{"attributes":{"ram_gb":{"value":"16","confidence":0.95},'
                    '"storage_gb":{"value":null,"confidence":0.0}},'
                    '"unsupported_facts":[]}'
                ),
                model="fake-model",
                prompt_tokens=1,
                completion_tokens=1,
            )
        ]
    )

    result = extract_attributes(product, specs, client)

    assert result.attributes["ram_gb"].value == "16"
    assert result.attributes["storage_gb"].value is None