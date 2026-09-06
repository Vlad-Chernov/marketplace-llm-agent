# Review Clustering Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Сравнить локальную кластеризацию отзывов по E5-эмбеддингам с LLM-классификацией по фиксированной таксономии на воспроизводимом наборе из 200 отзывов.

**Architecture:** Модуль кластеризации получает очищенные отзывы и локальные E5-векторы; он не вызывает LLM и не читает таксономию. Evaluator запускает таксономический и кластерный пути на одном очищенном наборе, применяет скрытые метки только после обработки для offline-метрик и отдаёт безопасный результат CLI.

**Tech Stack:** Python 3.13, Pydantic v2, sentence-transformers `intfloat/multilingual-e5-small`, pytest, Ruff, GigaChat.

**Spec:** `docs/superpowers/specs/2026-09-06-review-clustering-comparison-design.md`

## Global Constraints

- Данные: `generate_clean_products(count=10, seed=7)`, затем `generate_reviews(products, count=200, seed=42)`.
- `clean_reviews` вызывается один раз перед обеими ветками; фактическое число отзывов может стать меньше 200 из-за дублей и шума.
- `defect_label` и `is_delivery_review` не передаются рабочим алгоритмам и не сохраняются в JSON.
- Кластеризация: E5 локально, cosine threshold `0.82`, сортировка по `review_id`, ID `CLUSTER-001`.
- JSON не содержит тексты, PII, промпты, ответы LLM, скрытые метки или сопоставление кластера со скрытой меткой.
- Живой запуск использует `LLM_PROGRESS=1`; 200 LLM-вызовов могут выполняться долго.
- Не добавлять личный план `docs/superpowers/plans/2026-09-05-data-scaling.md` и `.docx`.

---

## File Structure

- Create `src/marketplace_agent/reviews/clustering.py` — кластеризация и модели кластеров.
- Create `src/marketplace_agent/evals/review_clustering.py` — offline-метрики, runner и безопасная сериализация.
- Create `evals/compare_review_analysis.py` — CLI и JSON-артефакт.
- Create `tests/reviews/test_clustering.py` — unit-тесты кластеризации.
- Create `tests/evals/test_review_clustering.py` — метрики, runner и безопасность JSON.
- Do not modify production `reviews/analyzer.py`, `reviews/analysis.py` и `review_metrics.py`.

---

### Task 1: Детерминированная кластеризация

**Files:**

- Create: `src/marketplace_agent/reviews/clustering.py`
- Test: `tests/reviews/test_clustering.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class ReviewCluster:
    cluster_id: str
    review_ids: list[str]

def cluster_review_vectors(
    reviews: Sequence[Review],
    vectors: Sequence[Sequence[float]],
    threshold: float = 0.82,
) -> list[ReviewCluster]:
    """Return deterministic clusters."""
```

- [ ] **Step 1: Write the failing tests**

```python
from datetime import date

import pytest

from marketplace_agent.domain.models import Review
from marketplace_agent.reviews.clustering import cluster_review_vectors


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
        [review("REV-000002"), review("REV-000001"), review("REV-000003")],
        [[0.0, 1.0], [1.0, 0.0], [0.99, 0.01]],
        threshold=0.90,
    )

    assert [
        (cluster.cluster_id, cluster.review_ids) for cluster in result
    ] == [
        ("CLUSTER-001", ["REV-000001", "REV-000003"]),
        ("CLUSTER-002", ["REV-000002"]),
    ]


def test_rejects_invalid_threshold_and_vector_shape() -> None:
    with pytest.raises(ValueError, match=r"threshold must be in \(0, 1\]"):
        cluster_review_vectors([review("REV-000001")], [[1.0, 0.0]], 0.0)

    with pytest.raises(
        ValueError,
        match="All vectors must have the same dimension",
    ):
        cluster_review_vectors(
            [review("REV-000001"), review("REV-000002")],
            [[1.0, 0.0], [1.0]],
        )
```

- [ ] **Step 2: Verify failure**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/reviews/test_clustering.py -v
```

Expected: collection error because the module is absent.

- [ ] **Step 3: Implement the minimal algorithm**

Create the module. Sort paired `(review, vector)` values by `review.review_id`. Normalize every vector. For every review, calculate cosine similarity to each normalized cluster centroid; add it to the highest-similarity cluster only if its score is at least the threshold, otherwise create a cluster. Recalculate the centroid as the normalized sum after an addition.

Use these exact validation errors:

```python
if not 0.0 < threshold <= 1.0:
    raise ValueError("threshold must be in (0, 1]")
if len(reviews) != len(vectors):
    raise ValueError("Review count must match vector count.")
if not vector:
    raise ValueError("Vectors must not be empty.")
if norm == 0.0:
    raise ValueError("Vectors must have a non-zero norm.")
if len({len(vector) for vector in normalized_vectors}) != 1:
    raise ValueError("All vectors must have the same dimension.")
```

Keep centroids private. Return only cluster ID and review IDs.

- [ ] **Step 4: Verify and commit**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/reviews/test_clustering.py -v
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run ruff check --fix \
  src/marketplace_agent/reviews/clustering.py \
  tests/reviews/test_clustering.py
git diff --check
git add src/marketplace_agent/reviews/clustering.py \
  tests/reviews/test_clustering.py
git commit -m "feat: cluster review embeddings deterministically"
```

---

### Task 2: Offline-метрики кластеров

**Files:**

- Create: `src/marketplace_agent/evals/review_clustering.py`
- Test: `tests/evals/test_review_clustering.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class ReviewClusterEvaluation:
    weighted_purity: float
    defect_recall: float
    mean_absolute_frequency_error: float
    noise_share: float
    cluster_count: int
    largest_cluster_size: int

def evaluate_review_clusters(
    reviews: Sequence[Review],
    clusters: Sequence[ReviewCluster],
) -> ReviewClusterEvaluation:
    """Return offline metrics for review clusters."""
```

- [ ] **Step 1: Write the failing metrics test**

```python
from datetime import date

from marketplace_agent.domain.models import Review
from marketplace_agent.evals.review_clustering import evaluate_review_clusters
from marketplace_agent.reviews.clustering import ReviewCluster


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
            review("REV-000004", None, delivery=True),
        ],
        [
            ReviewCluster("CLUSTER-001", ["REV-000001", "REV-000002"]),
            ReviewCluster("CLUSTER-002", ["REV-000003", "REV-000004"]),
        ],
    )

    assert result.weighted_purity == 0.75
    assert result.defect_recall == 1.0
    assert result.mean_absolute_frequency_error == 0.5
    assert result.noise_share == 0.0
    assert result.cluster_count == 2
    assert result.largest_cluster_size == 2
```

- [ ] **Step 2: Verify failure**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/evals/test_review_clustering.py::test_evaluates_cluster_metrics -v
```

Expected: import error because the evaluator module is absent.

- [ ] **Step 3: Implement metrics**

Use this private function only in the evaluator:

```python
def _evaluation_label(review: Review) -> str:
    if review.is_delivery_review:
        return "delivery_or_service"
    return review.defect_label or "no_defect"
```

Resolve equal label counts alphabetically. For each cluster, purity is its dominant-label count divided by cluster size. Weighted purity is the sum of dominant-label counts divided by all clustered reviews. Defect recall is the fraction of distinct hidden defect labels that dominate at least one cluster. Add an entire cluster size to a defect's predicted count only when the dominant label is a hidden defect; MAE compares those counts with the real defect frequencies. Noise share counts reviews in clusters dominated by `no_defect` or `delivery_or_service`.

Raise:

```python
ValueError(f"Cluster references an unknown review ID: {review_id}")
ValueError("Review IDs must belong to one cluster.")
```

- [ ] **Step 4: Verify and commit**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/evals/test_review_clustering.py -v
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run ruff check --fix \
  src/marketplace_agent/evals/review_clustering.py \
  tests/evals/test_review_clustering.py
git diff --check
git add src/marketplace_agent/evals/review_clustering.py \
  tests/evals/test_review_clustering.py
git commit -m "feat: measure review embedding clusters"
```

---

### Task 3: Runner двух сравниваемых путей

**Files:**

- Modify: `src/marketplace_agent/evals/review_clustering.py`
- Modify: `tests/evals/test_review_clustering.py`

**Interfaces:**

```python
class ReviewEmbedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Create one embedding for every review text."""

@dataclass(frozen=True)
class ReviewClusteringExperimentResult:
    taxonomy_evaluation: ReviewClassificationEvaluation
    cluster_evaluation: ReviewClusterEvaluation
    clusters: list[ReviewCluster]
    cleaned_review_count: int
    taxonomy_latency_ms: int
    clustering_latency_ms: int
    taxonomy_error_types: dict[str, int]

def run_review_clustering_experiment(
    reviews: Sequence[Review],
    taxonomy: DefectTaxonomy,
    llm: LLMClient,
    embedder: ReviewEmbedder,
    similarity_threshold: float = 0.82,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> ReviewClusteringExperimentResult:
    """Run both evaluation paths on one cleaned dataset."""
```

- [ ] **Step 1: Write the failing runner test**

```python
class FakeEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        assert texts == ["Перегрев.", "Батарея."]
        return [[1.0, 0.0], [0.0, 1.0]]


def test_runs_both_paths_on_same_cleaned_reviews() -> None:
    result = run_review_clustering_experiment(
        reviews=[
            review("REV-000001", "overheating").model_copy(
                update={"text": "Перегрев."}
            ),
            review("REV-000002", "battery_drain").model_copy(
                update={"text": "Батарея."}
            ),
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
```

Define local `taxonomy()` with categories `overheating`, `battery_drain` and `no_defect`. Define `response(defect_id, severity)` to return `LLMResponse` whose JSON has `kind="product_defect"` and `confidence=0.9`.

- [ ] **Step 2: Verify failure**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/evals/test_review_clustering.py::test_runs_both_paths_on_same_cleaned_reviews -v
```

Expected: import error because the runner is absent.

- [ ] **Step 3: Implement runner**

1. Run `clean_reviews(list(reviews))` once; pass its `.reviews` to both paths.
2. Time taxonomy classification with `perf_counter`. Before each request, emit `taxonomy_started` with `position`, `total` and `review_id`; catch `Exception`, count only its class name, emit `taxonomy_error`, and continue.
3. Call `evaluate_review_classification` with successful labels.
4. Time `embedder.embed_documents([review.text for review in cleaned_reviews])` and `cluster_review_vectors`.
5. If the number of vectors differs, raise `ValueError("Embedding count must match cleaned review count.")`.
6. Call `cluster_review_vectors` with `threshold=similarity_threshold`, then call `evaluate_review_clusters` and return the immutable result.

The clustering call must receive no taxonomy, labels or hidden fields.

- [ ] **Step 4: Verify and commit**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/evals/test_review_clustering.py \
  tests/reviews/test_clustering.py \
  tests/evals/test_review_metrics.py -v
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run ruff check --fix \
  src/marketplace_agent/evals/review_clustering.py \
  tests/evals/test_review_clustering.py
git diff --check
git add src/marketplace_agent/evals/review_clustering.py \
  tests/evals/test_review_clustering.py
git commit -m "feat: run review clustering comparison"
```

---

### Task 4: Безопасный CLI и JSON

**Files:**

- Create: `evals/compare_review_analysis.py`
- Modify: `src/marketplace_agent/evals/review_clustering.py`
- Modify: `tests/evals/test_review_clustering.py`

**Interfaces:**

```python
def serialize_review_clustering_result(
    result: ReviewClusteringExperimentResult,
) -> dict[str, object]:
    """Return only safe experiment fields."""
```

- [ ] **Step 1: Write the failing serialization test**

```python
def test_serializes_clusters_without_review_text_or_hidden_labels() -> None:
    payload = serialize_review_clustering_result(experiment_result())

    assert payload["clusters"] == [
        {"cluster_id": "CLUSTER-001", "review_ids": ["REV-000001"]}
    ]
    assert "text" not in repr(payload)
    assert "defect_label" not in repr(payload)
    assert "expected_defect_id" not in repr(payload)
```

Place an `experiment_result()` fixture in this test file. Construct it from metric dataclasses and `ReviewCluster`, never from a `Review` object.

- [ ] **Step 2: Verify failure**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/evals/test_review_clustering.py::test_serializes_clusters_without_review_text_or_hidden_labels -v
```

Expected: import error because serializer is absent.

- [ ] **Step 3: Implement serializer and CLI**

Serializer returns only count, both latency fields, error-type counts, taxonomy aggregate metrics excluding `.errors`, all cluster aggregate metrics and pairs of `cluster_id` / `review_ids`.

The CLI accepts:

```text
--review-count 200
--similarity-threshold 0.82
```

Reject every review count other than 200 with:

```python
raise ValueError("review-count must be 200 for E-005.")
```

The CLI creates data with fixed seeds, loads `data/taxonomy/laptop_defects.yaml`, instantiates `SentenceTransformerEmbedder`, and calls the runner with `similarity_threshold=arguments.similarity_threshold`. It passes a progress function only if `LLM_PROGRESS=1`.

Write `evals/runs/review-clustering-<run_id>.json` with run ID, UTC timestamp, both seeds, raw dataset SHA-256, E5 model name, threshold and serialized result. Compute the hash before cleaning from canonical `json.dumps([review.model_dump() for review in reviews], sort_keys=True, ensure_ascii=False)`; write only its SHA-256, never the source JSON. Print cleaned count, taxonomy recall, cluster recall, both latencies and output path.

- [ ] **Step 4: Verify and commit**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/evals/test_review_clustering.py \
  tests/reviews/test_clustering.py -v
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run python \
  evals/compare_review_analysis.py --help
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run ruff check --fix \
  src/marketplace_agent/reviews/clustering.py \
  src/marketplace_agent/evals/review_clustering.py \
  evals/compare_review_analysis.py \
  tests/reviews/test_clustering.py \
  tests/evals/test_review_clustering.py
git diff --check
git add src/marketplace_agent/reviews/clustering.py \
  src/marketplace_agent/evals/review_clustering.py \
  evals/compare_review_analysis.py \
  tests/reviews/test_clustering.py \
  tests/evals/test_review_clustering.py
git commit -m "feat: compare review clustering and taxonomy"
```

---

### Task 5: Живой эксперимент и E-005

**Files:**

- Modify: `EXPERIMENTS.md`
- Modify: `ROADMAP.md`
- Create locally: `evals/runs/review-clustering-<run_id>.json`

- [ ] **Step 1: Verify the project**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache make check
git diff --check
```

- [ ] **Step 2: Run the live experiment**

```bash
LLM_PROGRESS=1 \
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache \
uv run python evals/compare_review_analysis.py \
  --review-count 200 \
  --similarity-threshold 0.82
```

- [ ] **Step 3: Record the decision**

Add E-005 with run ID, provider/model, E5 model, threshold, seeds, hash, cleaned count, both metric rows, errors and the synthetic-data limitation. Mark 13.2 complete. Keep fixed taxonomy as production reporting unless weighted purity is at least 0.80 and clusters do not materially mix defects with delivery or no-defect reviews.

- [ ] **Step 4: Final verification and commit**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache make check
git diff --check
git add EXPERIMENTS.md ROADMAP.md
git commit -m "evals: compare review clustering and taxonomy"
git status --short
```
