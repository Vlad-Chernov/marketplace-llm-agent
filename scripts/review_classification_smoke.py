from datetime import date
from pathlib import Path

from marketplace_agent.config import Settings
from marketplace_agent.domain.models import Review
from marketplace_agent.llm.factory import create_llm_client
from marketplace_agent.reviews.analyzer import classify_review
from marketplace_agent.reviews.taxonomy import load_defect_taxonomy


def main() -> None:
    """Classify one real review through the configured LLM provider."""

    review = Review(
        review_id="REV-999999",
        sku="LAP-0001",
        rating=1,
        text="Ноутбук сильно греется и выключается во время работы.",
        created_at=date(2026, 8, 20),
        helpful_count=0,
    )
    taxonomy = load_defect_taxonomy(
        Path("data/taxonomy/laptop_defects.yaml")
    )
    labels = classify_review(
        review,
        taxonomy,
        create_llm_client(Settings.from_environment()),
    )

    print(labels.model_dump_json(indent=2))


if __name__ == "__main__":
    main()