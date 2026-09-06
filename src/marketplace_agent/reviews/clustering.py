from collections.abc import Sequence
from dataclasses import dataclass
from math import sqrt

from marketplace_agent.domain.models import Review


@dataclass(frozen=True)
class ReviewCluster:
    """Store one deterministic cluster of review IDs."""

    cluster_id: str
    review_ids: list[str]


@dataclass
class _ClusterState:
    """Store private mutable data while building one cluster."""

    review_ids: list[str]
    vector_sum: list[float]
    centroid: list[float]


def cluster_review_vectors(
    reviews: Sequence[Review],
    vectors: Sequence[Sequence[float]],
    threshold: float = 0.82,
) -> list[ReviewCluster]:
    """Group nearby review vectors using cosine similarity."""

    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    if len(reviews) != len(vectors):
        raise ValueError("Review count must match vector count.")
    if not reviews:
        return []

    normalized_vectors = [
        _normalize(vector)
        for vector in vectors
    ]
    if len({len(vector) for vector in normalized_vectors}) != 1:
        raise ValueError("All vectors must have the same dimension.")

    clusters: list[_ClusterState] = []
    pairs = sorted(
        zip(reviews, normalized_vectors, strict=True),
        key=lambda pair: pair[0].review_id,
    )

    for review, vector in pairs:
        cluster_index = _best_cluster_index(clusters, vector)

        if (
            cluster_index is None
            or _cosine(clusters[cluster_index].centroid, vector)
            < threshold
        ):
            clusters.append(
                _ClusterState(
                    review_ids=[review.review_id],
                    vector_sum=list(vector),
                    centroid=list(vector),
                )
            )
            continue

        cluster = clusters[cluster_index]
        cluster.review_ids.append(review.review_id)
        cluster.vector_sum = [
            current + added
            for current, added in zip(
                cluster.vector_sum,
                vector,
                strict=True,
            )
        ]
        cluster.centroid = _normalize(cluster.vector_sum)

    return [
        ReviewCluster(
            cluster_id=f"CLUSTER-{index:03d}",
            review_ids=cluster.review_ids,
        )
        for index, cluster in enumerate(clusters, start=1)
    ]


def _normalize(vector: Sequence[float]) -> list[float]:
    if not vector:
        raise ValueError("Vectors must not be empty.")

    values = [float(value) for value in vector]
    norm = sqrt(sum(value * value for value in values))
    if norm == 0.0:
        raise ValueError("Vectors must have a non-zero norm.")

    return [value / norm for value in values]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(
        left_value * right_value
        for left_value, right_value in zip(left, right, strict=True)
    )


def _best_cluster_index(
    clusters: Sequence[_ClusterState],
    vector: Sequence[float],
) -> int | None:
    if not clusters:
        return None

    return max(
        range(len(clusters)),
        key=lambda index: _cosine(
            clusters[index].centroid,
            vector,
        ),
    )