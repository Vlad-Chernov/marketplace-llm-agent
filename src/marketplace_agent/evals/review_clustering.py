from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from marketplace_agent.domain.models import Review
from marketplace_agent.evals.review_metrics import (
    ReviewClassificationEvaluation,
    evaluate_review_classification,
)
from marketplace_agent.llm.base import LLMClient
from marketplace_agent.reviews.analyzer import (
    ReviewLabels,
    classify_review,
)
from marketplace_agent.reviews.cleaning import clean_reviews
from marketplace_agent.reviews.clustering import (
    ReviewCluster,
    cluster_review_vectors,
)
from marketplace_agent.reviews.taxonomy import DefectTaxonomy


@dataclass(frozen=True)
class ReviewClusterEvaluation:
    """Store offline quality metrics for review clusters."""

    weighted_purity: float
    defect_recall: float
    mean_absolute_frequency_error: float
    noise_share: float
    cluster_count: int
    largest_cluster_size: int

class ReviewEmbedder(Protocol):
    """Create local embeddings for cleaned review texts."""

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Return one vector for every input text."""


@dataclass(frozen=True)
class ReviewClusteringExperimentResult:
    """Store results of taxonomy and clustering evaluation."""

    taxonomy_evaluation: ReviewClassificationEvaluation
    cluster_evaluation: ReviewClusterEvaluation
    clusters: list[ReviewCluster]
    cleaned_review_count: int
    taxonomy_latency_ms: int
    clustering_latency_ms: int
    taxonomy_error_types: dict[str, int]


def evaluate_review_clusters(
    reviews: Sequence[Review],
    clusters: Sequence[ReviewCluster],
) -> ReviewClusterEvaluation:
    """Evaluate clusters using hidden labels only after clustering."""

    reviews_by_id = {
        review.review_id: review
        for review in reviews
    }
    expected_counts = Counter(
        review.defect_label
        for review in reviews
        if review.defect_label is not None
    )
    assigned_review_ids: set[str] = set()
    predicted_counts: Counter[str] = Counter()
    recalled_defects: set[str] = set()
    dominant_count_total = 0
    noise_review_count = 0

    for cluster in clusters:
        cluster_reviews = []

        for review_id in cluster.review_ids:
            if review_id not in reviews_by_id:
                raise ValueError(
                    f"Cluster references an unknown review ID: {review_id}"
                )
            if review_id in assigned_review_ids:
                raise ValueError(
                    "Review IDs must belong to one cluster."
                )

            assigned_review_ids.add(review_id)
            cluster_reviews.append(reviews_by_id[review_id])

        label_counts = Counter(
            _evaluation_label(review)
            for review in cluster_reviews
        )
        dominant_label, dominant_count = min(
            label_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
        dominant_count_total += dominant_count

        if dominant_label in expected_counts:
            predicted_counts[dominant_label] += len(cluster_reviews)
            recalled_defects.add(dominant_label)

        if dominant_label in {
            "delivery_or_service",
            "no_defect",
        }:
            noise_review_count += len(cluster_reviews)

    if assigned_review_ids != set(reviews_by_id):
        raise ValueError("Review IDs must belong to one cluster.")

    clustered_review_count = len(assigned_review_ids)
    frequency_errors = [
        abs(predicted_counts[defect_id] - expected_count)
        for defect_id, expected_count in expected_counts.items()
    ]

    return ReviewClusterEvaluation(
        weighted_purity=(
            dominant_count_total / clustered_review_count
            if clustered_review_count
            else 0.0
        ),
        defect_recall=(
            len(recalled_defects) / len(expected_counts)
            if expected_counts
            else 0.0
        ),
        mean_absolute_frequency_error=(
            sum(frequency_errors) / len(frequency_errors)
            if frequency_errors
            else 0.0
        ),
        noise_share=(
            noise_review_count / clustered_review_count
            if clustered_review_count
            else 0.0
        ),
        cluster_count=len(clusters),
        largest_cluster_size=max(
            (
                len(cluster.review_ids)
                for cluster in clusters
            ),
            default=0,
        ),
    )


def _evaluation_label(review: Review) -> str:
    if review.is_delivery_review:
        return "delivery_or_service"

    return review.defect_label or "no_defect"

def run_review_clustering_experiment(
    reviews: Sequence[Review],
    taxonomy: DefectTaxonomy,
    llm: LLMClient,
    embedder: ReviewEmbedder,
    similarity_threshold: float = 0.82,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> ReviewClusteringExperimentResult:
    """Run taxonomy and local-clustering paths on one cleaned dataset."""

    cleaned_reviews = clean_reviews(list(reviews)).reviews
    labels_by_review_id: dict[str, ReviewLabels] = {}
    taxonomy_error_types: Counter[str] = Counter()

    taxonomy_started_at = perf_counter()

    for position, review in enumerate(
        cleaned_reviews,
        start=1,
    ):
        progress_event = {
            "event_type": "taxonomy_started",
            "position": position,
            "total": len(cleaned_reviews),
            "review_id": review.review_id,
        }
        if progress is not None:
            progress(progress_event)

        try:
            labels_by_review_id[review.review_id] = classify_review(
                review,
                taxonomy,
                llm,
            )
        except Exception as error:  # noqa: BLE001
            taxonomy_error_types[type(error).__name__] += 1
            if progress is not None:
                progress(
                    {
                        "event_type": "taxonomy_error",
                        "position": position,
                        "total": len(cleaned_reviews),
                        "review_id": review.review_id,
                    }
                )

    taxonomy_latency_ms = round(
        (perf_counter() - taxonomy_started_at) * 1000
    )
    taxonomy_evaluation = evaluate_review_classification(
        cleaned_reviews,
        labels_by_review_id,
    )

    clustering_started_at = perf_counter()

    if cleaned_reviews:
        vectors = embedder.embed_documents(
            [review.text for review in cleaned_reviews]
        )
        if len(vectors) != len(cleaned_reviews):
            raise ValueError(
                "Embedding count must match cleaned review count."
            )

        clusters = cluster_review_vectors(
            cleaned_reviews,
            vectors,
            threshold=similarity_threshold,
        )
    else:
        clusters = []

    clustering_latency_ms = round(
        (perf_counter() - clustering_started_at) * 1000
    )
    cluster_evaluation = evaluate_review_clusters(
        cleaned_reviews,
        clusters,
    )

    return ReviewClusteringExperimentResult(
        taxonomy_evaluation=taxonomy_evaluation,
        cluster_evaluation=cluster_evaluation,
        clusters=clusters,
        cleaned_review_count=len(cleaned_reviews),
        taxonomy_latency_ms=taxonomy_latency_ms,
        clustering_latency_ms=clustering_latency_ms,
        taxonomy_error_types=dict(
            sorted(taxonomy_error_types.items())
        ),
    )

def serialize_review_clustering_result(
    result: ReviewClusteringExperimentResult,
) -> dict[str, object]:
    """Return only fields safe for a persisted experiment artifact."""

    return {
        "cleaned_review_count": result.cleaned_review_count,
        "taxonomy_latency_ms": result.taxonomy_latency_ms,
        "clustering_latency_ms": result.clustering_latency_ms,
        "taxonomy_error_types": result.taxonomy_error_types,
        "taxonomy_metrics": {
            "overall_recall": (
                result.taxonomy_evaluation.overall_recall
            ),
            "recall_by_defect": (
                result.taxonomy_evaluation.recall_by_defect
            ),
            "mean_absolute_frequency_error": (
                result.taxonomy_evaluation.mean_absolute_frequency_error
            ),
            "weak_defects": (
                result.taxonomy_evaluation.weak_defects
            ),
        },
        "cluster_metrics": {
            "weighted_purity": (
                result.cluster_evaluation.weighted_purity
            ),
            "defect_recall": (
                result.cluster_evaluation.defect_recall
            ),
            "mean_absolute_frequency_error": (
                result.cluster_evaluation.mean_absolute_frequency_error
            ),
            "noise_share": result.cluster_evaluation.noise_share,
            "cluster_count": result.cluster_evaluation.cluster_count,
            "largest_cluster_size": (
                result.cluster_evaluation.largest_cluster_size
            ),
        },
        "clusters": [
            {
                "cluster_id": cluster.cluster_id,
                "review_ids": cluster.review_ids,
            }
            for cluster in result.clusters
        ],
    }