from pathlib import Path

import yaml

from marketplace_agent.domain.models import AttributeSpec


def load_attribute_specs(path: Path) -> list[AttributeSpec]:
    """Load and validate category attribute specifications from YAML."""

    data = yaml.safe_load(path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise TypeError("Specification file must contain a YAML mapping.")

    raw_attributes = data.get("attributes")
    if not isinstance(raw_attributes, list):
        raise TypeError("Specification file must contain an attributes list.")

    return [AttributeSpec.model_validate(item) for item in raw_attributes]