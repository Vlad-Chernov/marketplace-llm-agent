# Hybrid Policy Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Объединить BM25 и vector через RRF и измерить Recall@5 всех трёх ретриверов на общем наборе запросов.

**Architecture:** `HybridRetriever` объединяет только позиции результатов и использует существующий `SearchResult`. Eval-модуль хранит воспроизводимые кейсы и результаты по каждому запросу; скрипт запускает все ретриверы на одном корпусе и сохраняет JSON плюс Markdown-отчёт.

**Tech Stack:** Python 3.13, Pydantic v2, ChromaDB, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-21-hybrid-policy-retrieval-design.md`

## Global Constraints

- RRF-константа равна `60`; каждый ретривер получает `candidate_k = 10`.
- Все три ретривера используют одинаковые `query`, `filters`, `k` и золотые кейсы.
- Recall@5 считается отдельно от генерации ответа.
- Золотой набор хранится в Git и ссылается только на существующие `chunk_id`.
- Подробные результаты каждого запроса сохраняются в `evals/runs/`.
- Сначала пишется тест, затем минимальная реализация.
- Проверка проекта выполняется командой `make check`.

---

## File Structure

- Create: `src/marketplace_agent/retrieval/hybrid.py` — RRF-слияние.
- Create: `data/gold/policy_retrieval_cases.json` — 12 вручную проверенных запросов.
- Create: `src/marketplace_agent/evals/retrieval_metrics.py` — модели кейсов и Recall@k.
- Create: `evals/evaluate_retrieval.py` — реальный сравнительный прогон.
- Modify: `evals/REPORT.md`, `Makefile`.
- Create: `tests/retrieval/test_hybrid.py`, `tests/evals/test_retrieval_dataset.py`, `tests/evals/test_retrieval_metrics.py`.

### Task 1: Add RRF hybrid retrieval

**Files:**

- Create: `src/marketplace_agent/retrieval/hybrid.py`
- Create: `tests/retrieval/test_hybrid.py`

**Interfaces:**

- Consumes: objects with `search(query, k, filters=None) -> list[SearchResult]`.
- Produces: `HybridRetriever.search(query, k, filters=None) -> list[SearchResult]`.
- Used later by: retrieval evaluation and support answers.

- [ ] **Step 1: Write failing tests**

```python
from marketplace_agent.retrieval.documents import PolicyChunk
from marketplace_agent.retrieval.hybrid import HybridRetriever
from marketplace_agent.retrieval.lexical import SearchResult


def result(chunk_id: str, rank: int) -> SearchResult:
    return SearchResult(
        chunk=PolicyChunk(
            chunk_id=chunk_id,
            document_id="policy",
            policy_type="policy",
            heading=chunk_id,
            text=chunk_id,
        ),
        score=1.0,
        rank=rank,
    )


class StubRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.calls: list[tuple[str, int, dict[str, str] | None]] = []

    def search(
        self,
        query: str,
        k: int,
        filters: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        self.calls.append((query, k, filters))
        return self._results


def test_combines_ranks_with_reciprocal_rank_fusion() -> None:
    lexical = StubRetriever([result("a", 1), result("b", 2)])
    vector = StubRetriever([result("b", 1), result("c", 2)])
    retriever = HybridRetriever(lexical, vector)

    results = retriever.search("вопрос", k=3)

    assert [item.chunk.chunk_id for item in results] == ["b", "a", "c"]
    assert results[0].score == 1 / 62 + 1 / 61
    assert [item.rank for item in results] == [1, 2, 3]


def test_passes_identical_filters_to_both_retrievers() -> None:
    lexical = StubRetriever([result("a", 1)])
    vector = StubRetriever([result("a", 1)])
    retriever = HybridRetriever(lexical, vector, candidate_k=10)

    results = retriever.search(
        "вопрос",
        k=1,
        filters={"document_id": "returns"},
    )

    assert results[0].chunk.chunk_id == "a"
    assert lexical.calls == [("вопрос", 10, {"document_id": "returns"})]
    assert vector.calls == [("вопрос", 10, {"document_id": "returns"})]
```

- [ ] **Step 2: Run tests and confirm the expected failure**

```bash
uv run pytest tests/retrieval/test_hybrid.py -v
```

Expected: collection error because `marketplace_agent.retrieval.hybrid` does not exist.

- [ ] **Step 3: Implement `HybridRetriever`**

Create a `Retriever` protocol with `search(query, k, filters=None)`. Create `HybridRetriever` with constructor parameters `lexical`, `vector`, `rrf_k=60`, `candidate_k=10`.

For both underlying result lists, sum `1 / (rrf_k + result.rank)` by `result.chunk.chunk_id`. Keep the associated chunk. Sort by descending RRF score, then `chunk_id`. Return at most `k` fresh `SearchResult` objects with ranks beginning at one. Empty query or non-positive `k` returns `[]`.

- [ ] **Step 4: Run focused tests**

```bash
uv run pytest tests/retrieval/test_hybrid.py -v
```

Expected: two passing tests.

### Task 2: Add a fixed retrieval dataset and metrics

**Files:**

- Create: `data/gold/policy_retrieval_cases.json`
- Create: `src/marketplace_agent/evals/retrieval_metrics.py`
- Create: `tests/evals/test_retrieval_dataset.py`
- Create: `tests/evals/test_retrieval_metrics.py`

**Interfaces:**

- Produces: `RetrievalCase`, `RetrievalEvaluation`, `evaluate_retriever` and `append_retrieval_report`.
- Used later by: `evals/evaluate_retrieval.py`.

- [ ] **Step 1: Create the manual dataset**

Add twelve cases, one per source chunk, with these exact IDs and expected chunk IDs:

```json
[
  {"id": "policy-001", "query": "Где посмотреть статус доставки?", "expected_chunk_ids": ["delivery-01"]},
  {"id": "policy-002", "query": "Товар повреждён при получении", "expected_chunk_ids": ["delivery-02"]},
  {"id": "policy-003", "query": "Почему поддержка не говорит детали чужого заказа?", "expected_chunk_ids": ["delivery-03"]},
  {"id": "policy-004", "query": "Можно ли обменять товар на аналогичную модель?", "expected_chunk_ids": ["exchange-01"]},
  {"id": "policy-005", "query": "Какие условия для обмена исправного ноутбука?", "expected_chunk_ids": ["exchange-02"]},
  {"id": "policy-006", "query": "Что будет, если похожего товара нет на складе?", "expected_chunk_ids": ["exchange-03"]},
  {"id": "policy-007", "query": "Сколько дней можно вернуть ноутбук?", "expected_chunk_ids": ["returns-01"]},
  {"id": "policy-008", "query": "В каком состоянии должен быть товар для возврата?", "expected_chunk_ids": ["returns-02"]},
  {"id": "policy-009", "query": "Можно ли вернуть товар с производственным дефектом?", "expected_chunk_ids": ["returns-03"]},
  {"id": "policy-010", "query": "Где указан гарантийный срок?", "expected_chunk_ids": ["warranty-01"]},
  {"id": "policy-011", "query": "Какие неисправности покрывает гарантия?", "expected_chunk_ids": ["warranty-02"]},
  {"id": "policy-012", "query": "Покрывает ли гарантия механические повреждения?", "expected_chunk_ids": ["warranty-03"]}
]
```

- [ ] **Step 2: Write failing tests**

```python
import json
from pathlib import Path

from marketplace_agent.evals.retrieval_metrics import RetrievalCase
from marketplace_agent.retrieval.documents import load_policy_chunks


def test_dataset_references_existing_policy_chunks() -> None:
    raw_cases = json.loads(
        Path("data/gold/policy_retrieval_cases.json").read_text(
            encoding="utf-8"
        )
    )
    cases = [RetrievalCase.model_validate(raw_case) for raw_case in raw_cases]
    chunk_ids = {
        chunk.chunk_id
        for chunk in load_policy_chunks(Path("data/support"))
    }

    assert len(cases) == 12
    assert {case.id for case in cases} == {
        f"policy-{index:03d}" for index in range(1, 13)
    }
    assert all(
        set(case.expected_chunk_ids) <= chunk_ids for case in cases
    )
```

```python
from marketplace_agent.evals.retrieval_metrics import (
    RetrievalCase,
    evaluate_retriever,
)
from marketplace_agent.retrieval.documents import PolicyChunk
from marketplace_agent.retrieval.lexical import SearchResult


class StubRetriever:
    def search(
        self,
        query: str,
        k: int,
        filters: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        return [
            SearchResult(
                chunk=PolicyChunk(
                    chunk_id="returns-01",
                    document_id="returns",
                    policy_type="returns",
                    heading="Возврат",
                    text="Текст",
                ),
                score=1.0,
                rank=1,
            )
        ]


def test_calculates_mean_recall_at_five_per_query() -> None:
    cases = [
        RetrievalCase(
            id="one",
            query="Первый",
            expected_chunk_ids=["returns-01"],
        ),
        RetrievalCase(
            id="two",
            query="Второй",
            expected_chunk_ids=["returns-01", "warranty-01"],
        ),
    ]

    evaluation = evaluate_retriever(StubRetriever(), cases, k=5)

    assert evaluation.recall_at_k == 0.75
    assert [result.recall_at_k for result in evaluation.results] == [1.0, 0.5]
```

- [ ] **Step 3: Run tests and confirm the expected failure**

```bash
uv run pytest tests/evals/test_retrieval_dataset.py tests/evals/test_retrieval_metrics.py -v
```

Expected: import error because `marketplace_agent.evals.retrieval_metrics` does not exist.

- [ ] **Step 4: Implement metrics**

Define Pydantic models:

- `RetrievalCase(id, query, expected_chunk_ids, filters=None)`;
- `RetrievalCaseResult(case_id, expected_chunk_ids, retrieved_chunk_ids, recall_at_k)`;
- `RetrievalEvaluation(recall_at_k, results)`.

`evaluate_retriever(retriever, cases, k=5)` calls every case exactly once with its filters. Per-case recall equals the intersection of expected and top-k retrieved IDs divided by expected IDs. The evaluation recall is the arithmetic mean of per-case recall, or `0.0` for no cases.

`append_retrieval_report(evaluations, report_path, k)` accepts a mapping of retriever names to evaluations and appends a Markdown section containing a Recall@k table plus a row for each case with missing chunk IDs.

- [ ] **Step 5: Run focused tests**

```bash
uv run pytest tests/evals/test_retrieval_dataset.py tests/evals/test_retrieval_metrics.py -v
```

Expected: two passing tests.

### Task 3: Run and persist retrieval comparison

**Files:**

- Create: `evals/evaluate_retrieval.py`
- Modify: `Makefile`
- Modify: `evals/REPORT.md`

**Interfaces:**

- Consumes: policy corpus, Chroma index, retrieval dataset and all three retrievers.
- Produces: one JSON result file and one Markdown report section.

- [ ] **Step 1: Create evaluation script**

The script loads the 12 JSON cases with `RetrievalCase.model_validate`, then builds:

```python
chunks = load_policy_chunks(PROJECT_ROOT / "data" / "support")
embedder = SentenceTransformerEmbedder()
bm25 = BM25Retriever(chunks)
vector = VectorRetriever.open(
    PROJECT_ROOT / "data" / "vector_store",
    embedder,
)
hybrid = HybridRetriever(bm25, vector)
```

It calls `evaluate_retriever` for `bm25`, `vector` and `hybrid` with `k=5`, saves all `model_dump()` results to a UUID-named JSON file under `evals/runs/`, calls `append_retrieval_report`, then prints each Recall@5 and the result path.

- [ ] **Step 2: Add Makefile target and lint coverage**

Append `evals/evaluate_retrieval.py` to both existing Ruff command lines.

Add this target:

```makefile
evaluate-retrieval:
	uv run python evals/evaluate_retrieval.py
```

Add `evaluate-retrieval` to `.PHONY`.

- [ ] **Step 3: Run the real comparison**

```bash
make ingest
make evaluate-retrieval
```

Expected: the index contains 12 chunks; the script prints Recall@5 for `bm25`, `vector` and `hybrid`; a per-query JSON file is created under `evals/runs/`; `evals/REPORT.md` receives a comparison table.

- [ ] **Step 4: Run the project check**

```bash
make check
```

Expected: Ruff completes without errors and all tests pass.

- [ ] **Step 5: Commit**

```bash
git add ROADMAP.md Makefile data/gold/policy_retrieval_cases.json evals/REPORT.md evals/evaluate_retrieval.py src/marketplace_agent/retrieval/hybrid.py src/marketplace_agent/evals/retrieval_metrics.py tests/retrieval/test_hybrid.py tests/evals/test_retrieval_dataset.py tests/evals/test_retrieval_metrics.py
git commit -m "feat: add hybrid policy retrieval"
```

## Plan Self-Review

- Спецификация покрыта: RRF, общий набор, Recall@5, раздельные результаты и отчёт.
- Все метрики считаются кодом; LLM и реранкер не добавлены.
- Тесты RRF и метрик не требуют реальной модели или сети.
