from pathlib import Path

from marketplace_agent.data_generation.specs import load_attribute_specs

SPEC_PATH = Path("data/specs/laptops.yaml")
SNEAKERS_SPEC_PATH = Path("data/specs/sneakers.yaml")
KETTLES_SPEC_PATH = Path("data/specs/kettles.yaml")


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

def test_loads_twelve_unique_sneaker_attributes() -> None:
    specs = load_attribute_specs(SNEAKERS_SPEC_PATH)

    assert len(specs) == 12
    assert len({spec.key for spec in specs}) == 12


def test_sneaker_spec_has_four_required_attributes() -> None:
    specs = load_attribute_specs(SNEAKERS_SPEC_PATH)

    required_keys = {spec.key for spec in specs if spec.required}

    assert required_keys == {
        "size_eu",
        "upper_material",
        "closure_type",
        "season",
    }

def test_loads_twelve_unique_kettle_attributes() -> None:
    specs = load_attribute_specs(KETTLES_SPEC_PATH)

    assert len(specs) == 12
    assert len({spec.key for spec in specs}) == 12


def test_kettle_spec_has_four_required_attributes() -> None:
    specs = load_attribute_specs(KETTLES_SPEC_PATH)

    required_keys = {spec.key for spec in specs if spec.required}

    assert required_keys == {
        "capacity_l",
        "power_w",
        "body_material",
        "heating_element",
    }