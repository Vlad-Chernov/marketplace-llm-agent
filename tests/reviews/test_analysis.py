from datetime import date

from marketplace_agent.domain.models import Review
from marketplace_agent.reviews.analysis import analyze_reviews
from marketplace_agent.reviews.analyzer import ReviewLabels


def make_review(
    review_id: str,
    created_at: date,
) -> Review:
    return Review(
        review_id=review_id,
        sku="LAP-0001",
        rating=2,
        text="Тестовый отзыв.",
        created_at=created_at,
        helpful_count=0,
    )


def test_calculates_defect_frequency_share_and_growth() -> None:
    reviews = [
        make_review("REV-000001", date(2026, 1, 1)),
        make_review("REV-000002", date(2026, 1, 10)),
        make_review("REV-000003", date(2026, 2, 10)),
        make_review("REV-000004", date(2026, 2, 20)),
    ]
    labels_by_review_id = {
        "REV-000001": ReviewLabels(
            defect_id="no_defect",
            kind="no_defect",
            severity=None,
            confidence=0.9,
        ),
        "REV-000002": ReviewLabels(
            defect_id="no_defect",
            kind="delivery_or_service",
            severity=None,
            confidence=0.9,
        ),
        "REV-000003": ReviewLabels(
            defect_id="overheating",
            kind="product_defect",
            severity="high",
            confidence=0.9,
        ),
        "REV-000004": ReviewLabels(
            defect_id="overheating",
            kind="product_defect",
            severity="medium",
            confidence=0.9,
        ),
    }

    analysis = analyze_reviews(
        reviews=reviews,
        labels_by_review_id=labels_by_review_id,
        sku="LAP-0001",
        date_from=date(2026, 1, 1),
        date_to=date(2026, 2, 28),
    )

    overheating = analysis.defects[0]
    assert analysis.total_reviews == 4
    assert overheating.defect_id == "overheating"
    assert overheating.count == 2
    assert overheating.share == 0.5
    assert overheating.average_severity == 2.5
    assert overheating.trend == "increasing"
    assert overheating.example_review_ids == [
        "REV-000003",
        "REV-000004",
    ]