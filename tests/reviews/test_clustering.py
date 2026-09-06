from datetime import date

import pytest

from marketplace_agent.domain.models import Review
from marketplace_agent.reviews.clustering import (
    cluster_review_vectors,
)


def review(review_id: str) -> Review:
    return Review(
        review_id=review_id,
        sku="LAP-0001",
        rating=2,
        text="Тестовый отзыв.",
        created_at=date(2026, 1, 1),
        helpful_count=0,
    )


def test_clusters_near_vectors_in_review_id_order() -> None:
    result = cluster_review_vectors(
        [
            review("REV-000002"),
            review("REV-000001"),
            review("REV-000003"),
        ],
        [
            [0.0, 1.0],
            [1.0, 0.0],
            [0.99, 0.01],
        ],
        threshold=0.90,
    )

    assert [
        (cluster.cluster_id, cluster.review_ids)
        for cluster in result
    ] == [
        ("CLUSTER-001", ["REV-000001", "REV-000003"]),
        ("CLUSTER-002", ["REV-000002"]),
    ]


def test_rejects_invalid_threshold_and_vector_shape() -> None:
    with pytest.raises(
        ValueError,
        match=r"threshold must be in \(0, 1\]",
    ):
        cluster_review_vectors(
            [review("REV-000001")],
            [[1.0, 0.0]],
            0.0,
        )

    with pytest.raises(
        ValueError,
        match="All vectors must have the same dimension",
    ):
        cluster_review_vectors(
            [
                review("REV-000001"),
                review("REV-000002"),
            ],
            [
                [1.0, 0.0],
                [1.0],
            ],
        )