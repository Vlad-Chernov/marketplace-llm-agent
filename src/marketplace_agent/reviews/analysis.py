from collections import defaultdict
from collections.abc import Mapping
from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel, Field

from marketplace_agent.domain.models import Review
from marketplace_agent.reviews.analyzer import ReviewLabels

Trend = Literal["increasing", "stable", "decreasing"]

SEVERITY_VALUES = {
    "low": 1,
    "medium": 2,
    "high": 3,
}


class DefectAnalysis(BaseModel):
    """Store aggregate metrics for one product defect."""

    defect_id: str
    count: int = Field(ge=1)
    share: float = Field(ge=0.0, le=1.0)
    average_severity: float = Field(ge=1.0, le=3.0)
    trend: Trend
    example_review_ids: list[str]


class ReviewAnalysis(BaseModel):
    """Store deterministic analysis for one SKU and time period."""

    sku: str
    date_from: date
    date_to: date
    total_reviews: int = Field(ge=0)
    defects: list[DefectAnalysis]


def analyze_reviews(
    reviews: list[Review],
    labels_by_review_id: Mapping[str, ReviewLabels],
    sku: str,
    date_from: date,
    date_to: date,
) -> ReviewAnalysis:
    """Aggregate classified product defects for one SKU and period."""

    selected_reviews = [
        review
        for review in reviews
        if review.sku == sku and date_from <= review.created_at <= date_to
    ]
    reviews_by_defect: dict[str, list[Review]] = defaultdict(list)

    for review in selected_reviews:
        labels = labels_by_review_id.get(review.review_id)
        if labels is not None and labels.kind == "product_defect":
            reviews_by_defect[labels.defect_id].append(review)

    midpoint = date_from + timedelta(days=(date_to - date_from).days // 2)
    defects = [
        _build_defect_analysis(
            defect_id,
            defect_reviews,
            labels_by_review_id,
            len(selected_reviews),
            midpoint,
        )
        for defect_id, defect_reviews in reviews_by_defect.items()
    ]

    return ReviewAnalysis(
        sku=sku,
        date_from=date_from,
        date_to=date_to,
        total_reviews=len(selected_reviews),
        defects=sorted(defects, key=lambda defect: (-defect.count, defect.defect_id)),
    )


def _build_defect_analysis(
    defect_id: str,
    reviews: list[Review],
    labels_by_review_id: Mapping[str, ReviewLabels],
    total_reviews: int,
    midpoint: date,
) -> DefectAnalysis:
    severity_values = [
        SEVERITY_VALUES[labels_by_review_id[review.review_id].severity]
        for review in reviews
        if labels_by_review_id[review.review_id].severity is not None
    ]
    earlier_count = sum(review.created_at <= midpoint for review in reviews)
    later_count = len(reviews) - earlier_count

    return DefectAnalysis(
        defect_id=defect_id,
        count=len(reviews),
        share=len(reviews) / total_reviews,
        average_severity=sum(severity_values) / len(severity_values),
        trend=_trend(earlier_count, later_count),
        example_review_ids=[review.review_id for review in reviews[:3]],
    )


def _trend(earlier_count: int, later_count: int) -> Trend:
    if later_count > earlier_count:
        return "increasing"
    if later_count < earlier_count:
        return "decreasing"
    return "stable"