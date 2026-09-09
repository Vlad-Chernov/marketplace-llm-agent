from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from marketplace_agent.domain.models import Review
from marketplace_agent.reviews.analyzer import ReviewLabels

ReviewErrorType = Literal[
    "missed_defect",
    "wrong_defect",
    "delivery_as_product_defect",
    "false_product_defect",
]


@dataclass(frozen=True)
class ReviewEvaluationError:
    """Describe one incorrect review classification."""

    review_id: str
    expected_defect_id: str | None
    predicted_defect_id: str | None
    error_type: ReviewErrorType


@dataclass(frozen=True)
class ReviewClassificationEvaluation:
    """Store quality metrics for classified synthetic reviews."""

    overall_recall: float
    recall_by_defect: dict[str, float]
    mean_absolute_frequency_error: float
    weak_defects: list[str]
    errors: list[ReviewEvaluationError]
    defect_counts: dict[str, int] = field(default_factory=dict)


def evaluate_review_classification(
    reviews: list[Review],
    labels_by_review_id: Mapping[str, ReviewLabels],
) -> ReviewClassificationEvaluation:
    """Compare predicted labels with hidden synthetic evaluation labels."""

    expected_counts = Counter(
        review.defect_label
        for review in reviews
        if review.defect_label is not None
    )
    predicted_counts: Counter[str] = Counter()
    correct_by_defect: Counter[str] = Counter()
    errors: list[ReviewEvaluationError] = []

    for review in reviews:
        labels = labels_by_review_id.get(review.review_id)
        predicted_defect_id = (
            labels.defect_id
            if labels is not None and labels.kind == "product_defect"
            else "no_defect"
        )

        if (
            labels is not None
            and labels.kind == "product_defect"
            and predicted_defect_id in expected_counts
        ):
            predicted_counts[predicted_defect_id] += 1

        if review.is_delivery_review:
            if labels is not None and labels.kind == "product_defect":
                errors.append(
                    ReviewEvaluationError(
                        review_id=review.review_id,
                        expected_defect_id=None,
                        predicted_defect_id=predicted_defect_id,
                        error_type="delivery_as_product_defect",
                    )
                )
            continue

        if review.defect_label is None:
            if labels is not None and labels.kind == "product_defect":
                errors.append(
                    ReviewEvaluationError(
                        review_id=review.review_id,
                        expected_defect_id=None,
                        predicted_defect_id=predicted_defect_id,
                        error_type="false_product_defect",
                    )
                )
            continue

        if predicted_defect_id == review.defect_label:
            correct_by_defect[review.defect_label] += 1
        elif predicted_defect_id == "no_defect":
            errors.append(
                ReviewEvaluationError(
                    review_id=review.review_id,
                    expected_defect_id=review.defect_label,
                    predicted_defect_id=predicted_defect_id,
                    error_type="missed_defect",
                )
            )
        else:
            errors.append(
                ReviewEvaluationError(
                    review_id=review.review_id,
                    expected_defect_id=review.defect_label,
                    predicted_defect_id=predicted_defect_id,
                    error_type="wrong_defect",
                )
            )

    recall_by_defect = {
        defect_id: correct_by_defect[defect_id] / expected_count
        for defect_id, expected_count in sorted(expected_counts.items())
    }
    frequency_errors = [
        abs(predicted_counts[defect_id] - expected_count)
        for defect_id, expected_count in expected_counts.items()
    ]

    return ReviewClassificationEvaluation(
        overall_recall=(
            sum(correct_by_defect.values()) / sum(expected_counts.values())
            if expected_counts
            else 0.0
        ),
        recall_by_defect=recall_by_defect,
        mean_absolute_frequency_error=(
            sum(frequency_errors) / len(frequency_errors)
            if frequency_errors
            else 0.0
        ),
        weak_defects=[
            defect_id
            for defect_id, recall in recall_by_defect.items()
            if recall < 0.8
        ],
        errors=errors,
        defect_counts=dict(sorted(expected_counts.items())),
    )

def append_review_evaluation_report(
    evaluation: ReviewClassificationEvaluation,
    report_path: Path,
) -> None:
    """Append review-classification metrics and examples to Markdown."""

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("a", encoding="utf-8") as report_file:
        report_file.write("\n## Review classification evaluation\n\n")
        report_file.write(
            f"Overall recall: {evaluation.overall_recall:.3f}\n\n"
        )
        report_file.write(
            "Mean absolute frequency error: "
            f"{evaluation.mean_absolute_frequency_error:.3f}\n\n"
        )

        report_file.write("### Recall by defect\n\n")
        for defect_id, recall in evaluation.recall_by_defect.items():
            report_file.write(f"- {defect_id}: {recall:.3f}\n")

        report_file.write("\n### Weak defects\n\n")
        if evaluation.weak_defects:
            for defect_id in evaluation.weak_defects:
                report_file.write(f"- {defect_id}\n")
        else:
            report_file.write("- none\n")

        report_file.write("\n### Typical errors\n\n")
        if not evaluation.errors:
            report_file.write("- none\n")
            return

        report_file.write(
            "| Review ID | Expected | Predicted | Error type |\n"
        )
        report_file.write("| --- | --- | --- | --- |\n")
        for error in evaluation.errors:
            report_file.write(
                f"| {error.review_id} | {error.expected_defect_id} | "
                f"{error.predicted_defect_id} | {error.error_type} |\n"
            )
