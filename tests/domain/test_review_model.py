from datetime import date

import pytest
from pydantic import ValidationError

from marketplace_agent.domain.models import Review


def test_review_hides_evaluation_labels_in_json() -> None:
    review = Review(
        review_id="REV-000001",
        sku="LAP-0001",
        rating=2,
        text="Батарея слишком быстро разряжается.",
        created_at=date(2026, 1, 10),
        helpful_count=3,
        defect_label="battery_drain",
        contains_personal_data=False,
        is_delivery_review=False,
    )

    serialized = review.model_dump(mode="json")

    assert serialized["created_at"] == "2026-01-10"
    assert "defect_label" not in serialized
    assert "contains_personal_data" not in serialized
    assert "is_delivery_review" not in serialized


@pytest.mark.parametrize("rating", [0, 6])
def test_review_rejects_rating_outside_one_to_five(rating: int) -> None:
    with pytest.raises(ValidationError):
        Review(
            review_id="REV-000001",
            sku="LAP-0001",
            rating=rating,
            text="Текст отзыва.",
            created_at=date(2026, 1, 10),
            helpful_count=0,
        )