from pathlib import Path

from marketplace_agent.data_generation.reviews import DEFECT_TEXTS
from marketplace_agent.reviews.taxonomy import load_defect_taxonomy

TAXONOMY_PATH = Path("data/taxonomy/laptop_defects.yaml")


def test_taxonomy_covers_generated_defects_and_safe_classes() -> None:
    taxonomy = load_defect_taxonomy(TAXONOMY_PATH)
    category_ids = {category.id for category in taxonomy.categories}

    assert set(DEFECT_TEXTS) <= category_ids
    assert {"other", "no_defect"} <= category_ids
    assert len(category_ids) == len(taxonomy.categories)