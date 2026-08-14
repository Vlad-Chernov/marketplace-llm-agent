from pathlib import Path

from marketplace_agent.data_generation.specs import load_attribute_specs

SPEC_PATH = Path("data/specs/laptops.yaml")


def test_loads_twelve_unique_laptop_attributes() -> None:
    specs = load_attribute_specs(SPEC_PATH)

    assert len(specs) == 12
    assert len({spec.key for spec in specs}) == 12


def test_laptop_spec_has_four_required_attributes() -> None:
    specs = load_attribute_specs(SPEC_PATH)

    required_keys = {spec.key for spec in specs if spec.required}

    assert required_keys == {
        "screen_diagonal_in",
        "processor",
        "ram_gb",
        "storage_gb",
    }