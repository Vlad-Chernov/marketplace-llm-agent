from datetime import date

from marketplace_agent.domain.models import Review
from marketplace_agent.evals.review_clustering import (
    evaluate_review_clusters,
    run_review_clustering_experiment,
)
from marketplace_agent.llm.base import FakeLLMClient, LLMResponse
from marketplace_agent.reviews.clustering import ReviewCluster
from marketplace_agent.reviews.taxonomy import (
    DefectCategory,
    DefectTaxonomy,
)


def review(
    review_id: str,
    defect_label: str | None,
    *,
    delivery: bool = False,
) -> Review:
    return Review(
        review_id=review_id,
        sku="LAP-0001",
        rating=2,
        text="Тестовый отзыв.",
        created_at=date(2026, 1, 1),
        helpful_count=0,
        defect_label=defect_label,
        is_delivery_review=delivery,
    )


def test_evaluates_cluster_metrics() -> None:
    result = evaluate_review_clusters(
        [
            review("REV-000001", "overheating"),
            review("REV-000002", "overheating"),
            review("REV-000003", "battery_drain"),
            review(
                "REV-000004",
                None,
                delivery=True,
            ),
        ],
        [
            ReviewCluster(
                "CLUSTER-001",
                ["REV-000001", "REV-000002"],
            ),
            ReviewCluster(
                "CLUSTER-002",
                ["REV-000003", "REV-000004"],
            ),
        ],
    )

    assert result.weighted_purity == 0.75
    assert result.defect_recall == 1.0
    assert result.mean_absolute_frequency_error == 0.5
    assert result.noise_share == 0.0
    assert result.cluster_count == 2
    assert result.largest_cluster_size == 2

class FakeEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        assert texts == ["Перегрев.", "Батарея."]
        return [
            [1.0, 0.0],
            [0.0, 1.0],
        ]


def taxonomy() -> DefectTaxonomy:
    return DefectTaxonomy(
        categories=[
            DefectCategory(
                id="overheating",
                label="Перегрев",
                description="Нагрев.",
            ),
            DefectCategory(
                id="battery_drain",
                label="Батарея",
                description="Разряд.",
            ),
            DefectCategory(
                id="no_defect",
                label="Нет дефекта",
                description="Нет проблемы.",
            ),
        ]
    )


def response(
    defect_id: str,
    severity: str,
) -> LLMResponse:
    return LLMResponse(
        content=(
            "{"
            f'"defect_id":"{defect_id}",'
            '"kind":"product_defect",'
            f'"severity":"{severity}",'
            '"confidence":0.9'
            "}"
        ),
        model="fake-model",
        prompt_tokens=1,
        completion_tokens=1,
    )


def test_runs_both_paths_on_same_cleaned_reviews() -> None:
    result = run_review_clustering_experiment(
        reviews=[
            review(
                "REV-000001",
                "overheating",
            ).model_copy(update={"text": "Перегрев."}),
            review(
                "REV-000002",
                "battery_drain",
            ).model_copy(update={"text": "Батарея."}),
        ],
        taxonomy=taxonomy(),
        llm=FakeLLMClient(
            [
                response("overheating", "high"),
                response("battery_drain", "low"),
            ]
        ),
        embedder=FakeEmbedder(),
    )

    assert result.cleaned_review_count == 2
    assert result.taxonomy_evaluation.overall_recall == 1.0
    assert result.cluster_evaluation.defect_recall == 1.0
    assert [cluster.review_ids for cluster in result.clusters] == [
        ["REV-000001"],
        ["REV-000002"],
    ]