from datetime import date

from marketplace_agent.domain.models import Review
from marketplace_agent.evals.review_metrics import (
    ReviewClassificationEvaluation,
    ReviewEvaluationError,
    append_review_evaluation_report,
    evaluate_review_classification,
)
from marketplace_agent.reviews.analyzer import ReviewLabels


def make_review(
    review_id: str,
    defect_label: str | None,
    is_delivery_review: bool = False,
) -> Review:
    return Review(
        review_id=review_id,
        sku="LAP-0001",
        rating=2,
        text="Тестовый отзыв.",
        created_at=date(2026, 1, 1),
        helpful_count=0,
        defect_label=defect_label,
        is_delivery_review=is_delivery_review,
    )


def test_measures_recall_frequency_error_and_typical_errors() -> None:
    reviews = [
        make_review("REV-000001", "overheating"),
        make_review("REV-000002", "overheating"),
        make_review("REV-000003", "battery_drain"),
        make_review("REV-000004", None, is_delivery_review=True),
    ]
    labels_by_review_id = {
        "REV-000001": ReviewLabels(
            defect_id="overheating",
            kind="product_defect",
            severity="high",
            confidence=0.9,
        ),
        "REV-000002": ReviewLabels(
            defect_id="no_defect",
            kind="no_defect",
            severity=None,
            confidence=0.9,
        ),
        "REV-000003": ReviewLabels(
            defect_id="other",
            kind="product_defect",
            severity="low",
            confidence=0.9,
        ),
        "REV-000004": ReviewLabels(
            defect_id="overheating",
            kind="product_defect",
            severity="medium",
            confidence=0.9,
        ),
    }

    evaluation = evaluate_review_classification(
        reviews,
        labels_by_review_id,
    )

    assert evaluation.overall_recall == 1 / 3
    assert evaluation.recall_by_defect == {
        "battery_drain": 0.0,
        "overheating": 0.5,
    }
    assert evaluation.defect_counts == {
        "battery_drain": 1,
        "overheating": 2,
    }
    assert evaluation.mean_absolute_frequency_error == 0.5
    assert evaluation.weak_defects == [
        "battery_drain",
        "overheating",
    ]
    assert [
        (error.review_id, error.error_type)
        for error in evaluation.errors
    ] == [
        ("REV-000002", "missed_defect"),
        ("REV-000003", "wrong_defect"),
        ("REV-000004", "delivery_as_product_defect"),
    ]

def test_appends_review_evaluation_report(tmp_path) -> None:
    evaluation = ReviewClassificationEvaluation(
        overall_recall=0.5,
        recall_by_defect={"battery_drain": 0.0, "overheating": 1.0},
        mean_absolute_frequency_error=0.5,
        weak_defects=["battery_drain"],
        errors=[
            ReviewEvaluationError(
                review_id="REV-000001",
                expected_defect_id="battery_drain",
                predicted_defect_id="no_defect",
                error_type="missed_defect",
            )
        ],
    )
    report_path = tmp_path / "REPORT.md"

    append_review_evaluation_report(evaluation, report_path)

    report_text = report_path.read_text(encoding="utf-8")
    assert "Overall recall: 0.500" in report_text
    assert "battery_drain" in report_text
    assert "REV-000001" in report_text
    assert "missed_defect" in report_text
