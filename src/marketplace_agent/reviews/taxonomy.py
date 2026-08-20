from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DefectCategory(BaseModel):
    """Describe one fixed product-defect category."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    description: str = Field(min_length=1)
    examples: list[str] = Field(default_factory=list)


class DefectTaxonomy(BaseModel):
    """Store a fixed taxonomy for review classification."""

    model_config = ConfigDict(extra="forbid")

    categories: list[DefectCategory] = Field(min_length=1)


def load_defect_taxonomy(path: Path | str) -> DefectTaxonomy:
    """Load and validate a versioned defect taxonomy."""

    raw_data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return DefectTaxonomy.model_validate(raw_data)