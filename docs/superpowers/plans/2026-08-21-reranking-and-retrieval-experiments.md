# Reranking and Retrieval Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select a policy-retrieval configuration by measuring chunk size, vector/hybrid retrieval, and cross-encoder reranking.

**Architecture:** Build deterministic chunk-size variants independently of the existing document parser. Add a reranker that consumes already retrieved candidates. Extend evaluation records with reciprocal rank and latency, then run two controlled comparison groups and persist every query result.

**Tech Stack:** Python 3.13, Pydantic v2, pytest, Ruff, ChromaDB, sentence-transformers.

**Spec:** `docs/superpowers/specs/2026-08-21-reranking-and-retrieval-experiments-design.md`

## Global Constraints

- Keep BM25 and `VectorRetriever` implementations unchanged.
- Rerank only the hybrid top-10 candidates with `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.
- Use fake embedders and fake cross-encoder scorers in unit tests; tests must not download models.
- Chunk-size evaluation uses `document_id`; retrieval-strategy evaluation uses `chunk_id`.
- Persist per-query results to `evals/runs/` and write aggregate conclusions to `EXPERIMENTS.md`.
- Run `make check` before the final commit.

---

## File structure

- Create `src/marketplace_agent/retrieval/chunking.py`: deterministic small, medium, and large chunk variants.
- Create `src/marketplace_agent/retrieval/reranker.py`: reranker protocol and cross-encoder implementation.
- Modify `src/marketplace_agent/evals/retrieval_metrics.py`: identifier-aware Recall@k, MRR, and latency.
- Create `evals/compare_retrieval_experiments.py`: execute controlled experiment groups and persist JSON.
- Create `tests/retrieval/test_chunking.py`, `tests/retrieval/test_reranker.py`, and `tests/evals/test_retrieval_experiments.py`.
- Modify `tests/evals/test_retrieval_metrics.py`, `Makefile`, `EXPERIMENTS.md`, and `ROADMAP.md`.

### Task 1: Deterministic chunk-size variants

**Files:**
- Create: `tests/retrieval/test_chunking.py`
- Create: `src/marketplace_agent/retrieval/chunking.py`

**Interfaces:**
- Consumes: `PolicyChunk` from `marketplace_agent.retrieval.documents`.
- Produces: `build_chunk_variants(chunks: Sequence[PolicyChunk]) -> dict[str, list[PolicyChunk]]`.

- [ ] **Step 1: Write the failing test**

```python
from marketplace_agent.retrieval.chunking import build_chunk_variants
from marketplace_agent.retrieval.documents import PolicyChunk


def test_builds_unique_small_medium_and_large_variants() -> None:
    chunks = [
        PolicyChunk(chunk_id="returns-01", document_id="returns", policy_type="returns", heading="Возврат", text="Первый"),
        PolicyChunk(chunk_id="returns-02", document_id="returns", policy_type="returns", heading="Возврат", text="Второй"),
        PolicyChunk(chunk_id="returns-03", document_id="returns", policy_type="returns", heading="Возврат", text="Третий"),
    ]

    variants = build_chunk_variants(chunks)

    assert [chunk.chunk_id for chunk in variants["small"]] == ["small-returns-01", "small-returns-02", "small-returns-03"]
    assert [chunk.text for chunk in variants["medium"]] == ["Первый\n\nВторой", "Третий"]
    assert [chunk.text for chunk in variants["large"]] == ["Первый\n\nВторой\n\nТретий"]
    assert all(chunk.document_id == "returns" for chunks in variants.values() for chunk in chunks)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/retrieval/test_chunking.py -v`  
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
from collections import defaultdict
from collections.abc import Sequence

from marketplace_agent.retrieval.documents import PolicyChunk


def build_chunk_variants(chunks: Sequence[PolicyChunk]) -> dict[str, list[PolicyChunk]]:
    grouped: dict[str, list[PolicyChunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.document_id].append(chunk)

    return {
        "small": _small(chunks),
        "medium": _medium(grouped),
        "large": _large(grouped),
    }
```

Implement `_small`, `_medium`, and `_large` so IDs begin with the variant name, each original document is processed in source order, medium joins adjacent texts with `"\n\n"`, and large produces one chunk per document.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/retrieval/test_chunking.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/marketplace_agent/retrieval/chunking.py tests/retrieval/test_chunking.py
git commit -m "feat: build policy chunk size variants"
```

### Task 2: Cross-encoder reranker

**Files:**
- Create: `tests/retrieval/test_reranker.py`
- Create: `src/marketplace_agent/retrieval/reranker.py`

**Interfaces:**
- Consumes: `SearchResult` from `marketplace_agent.retrieval.lexical`.
- Produces: `Reranker.rerank(query: str, results: Sequence[SearchResult], k: int) -> list[SearchResult]` and `CrossEncoderReranker`.

- [ ] **Step 1: Write the failing test**

```python
def test_reranks_candidates_and_reassigns_ranks() -> None:
    reranker = CrossEncoderReranker(scorer=FakeScorer([0.2, 0.9]))
    results = [make_result("delivery-01", rank=1), make_result("returns-01", rank=2)]

    reranked = reranker.rerank("Как вернуть товар?", results, k=1)

    assert [result.chunk.chunk_id for result in reranked] == ["returns-01"]
    assert [(result.score, result.rank) for result in reranked] == [(0.9, 1)]


def test_returns_empty_without_scoring_invalid_request() -> None:
    scorer = FakeScorer([])
    assert CrossEncoderReranker(scorer=scorer).rerank("", [make_result("returns-01", 1)], 5) == []
    assert scorer.pairs == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/retrieval/test_reranker.py -v`  
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Write minimal implementation**

```python
class CrossEncoderReranker:
    def __init__(self, scorer: PairScorer | None = None) -> None:
        self._scorer = scorer or SentenceTransformerCrossEncoder()

    def rerank(self, query: str, results: Sequence[SearchResult], k: int) -> list[SearchResult]:
        if not query.strip() or k <= 0 or not results:
            return []
        scores = self._scorer.score([(query, result.chunk.text) for result in results])
        ranked = sorted(zip(results, scores, strict=True), key=lambda item: (-float(item[1]), item[0].chunk.chunk_id))
        return [SearchResult(chunk=result.chunk, score=float(score), rank=rank) for rank, (result, score) in enumerate(ranked[:k], start=1)]
```

`SentenceTransformerCrossEncoder` lazily creates `CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")` and exposes `score(pairs) -> list[float]` through `predict`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/retrieval/test_reranker.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/marketplace_agent/retrieval/reranker.py tests/retrieval/test_reranker.py
git commit -m "feat: rerank policy retrieval candidates"
```

### Task 3: Evaluation metrics for document relevance, MRR, and latency

**Files:**
- Modify: `src/marketplace_agent/evals/retrieval_metrics.py`
- Modify: `tests/evals/test_retrieval_metrics.py`

**Interfaces:**
- Consumes: `Retriever.search(query, k, filters)`.
- Produces: `evaluate_retriever(retriever, cases, k=5, relevance_key="chunk_id") -> RetrievalEvaluation` where `relevance_key` is `"chunk_id"` or `"document_id"`.

- [ ] **Step 1: Write the failing test**

```python
def test_calculates_mrr_latency_and_document_relevance(monkeypatch) -> None:
    monkeypatch.setattr("marketplace_agent.evals.retrieval_metrics.perf_counter", iter([1.0, 1.025]).__next__)
    case = RetrievalCase(id="one", query="Вернуть", expected_chunk_ids=["returns-01"])

    evaluation = evaluate_retriever(StubRetriever(), [case], k=5, relevance_key="document_id")

    assert evaluation.recall_at_k == 1.0
    assert evaluation.mrr == 1.0
    assert evaluation.mean_latency_ms == 25.0
    assert evaluation.results[0].retrieved_document_ids == ["returns"]
    assert evaluation.results[0].latency_ms == 25.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/evals/test_retrieval_metrics.py -v`  
Expected: FAIL because `relevance_key`, MRR, and latency fields do not exist.

- [ ] **Step 3: Write minimal implementation**

```python
started_at = perf_counter()
search_results = retriever.search(case.query, k, case.filters)
latency_ms = round((perf_counter() - started_at) * 1000, 3)
retrieved_ids = [getattr(result.chunk, relevance_key) for result in search_results[:k]]
expected_ids = _expected_ids(case, relevance_key)
first_rank = next((rank for rank, identifier in enumerate(retrieved_ids, start=1) if identifier in expected_ids), None)
reciprocal_rank = 0.0 if first_rank is None else 1 / first_rank
```

For `document_id`, derive expected IDs by removing the final `-NN` suffix from every expected chunk ID. Add fields `retrieved_document_ids`, `first_relevant_rank`, `reciprocal_rank`, and `latency_ms` to `RetrievalCaseResult`, and `mrr` plus `mean_latency_ms` to `RetrievalEvaluation`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/evals/test_retrieval_metrics.py tests/evals/test_retrieval_dataset.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/marketplace_agent/evals/retrieval_metrics.py tests/evals/test_retrieval_metrics.py
git commit -m "eval: measure retrieval MRR and latency"
```

### Task 4: Reproducible controlled experiment runner

**Files:**
- Create: `tests/evals/test_retrieval_experiments.py`
- Create: `evals/compare_retrieval_experiments.py`
- Modify: `Makefile`
- Modify: `EXPERIMENTS.md`

**Interfaces:**
- Consumes: chunk variants, `BM25Retriever`, `build_vector_index`, `VectorRetriever`, `HybridRetriever`, `CrossEncoderReranker`, and `evaluate_retriever`.
- Produces: one `evals/runs/retrieval-experiments-<uuid>.json` and command `make compare-retrieval`.

- [ ] **Step 1: Write the failing test**

```python
def test_builds_controlled_experiment_groups() -> None:
    groups = build_experiment_groups(fake_components())

    assert set(groups["chunk_size"]) == {"hybrid-small", "hybrid-medium", "hybrid-large"}
    assert set(groups["strategy"]) == {"vector-medium", "hybrid-medium", "hybrid-reranked-medium"}
    assert groups["chunk_size"]["hybrid-small"].relevance_key == "document_id"
    assert groups["strategy"]["hybrid-reranked-medium"].relevance_key == "chunk_id"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/evals/test_retrieval_experiments.py -v`  
Expected: FAIL with an import error.

- [ ] **Step 3: Write minimal implementation**

```python
def main() -> None:
    cases = load_cases()
    variants = build_chunk_variants(load_policy_chunks(PROJECT_ROOT / "data" / "support"))
    evaluations = run_experiments(cases, variants)
    results_path = PROJECT_ROOT / "evals" / "runs" / f"retrieval-experiments-{uuid4().hex}.json"
    results_path.write_text(json.dumps(evaluations, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Results: {results_path}")
```

Build an isolated temporary Chroma index for each chunk variant, then remove only that exact temporary directory after its evaluations complete. The chunk-size group evaluates hybrid with document relevance. The strategy group evaluates vector, hybrid, and a wrapper whose `search` calls hybrid with ten candidates then calls `reranker.rerank(query, candidates, k)` with chunk relevance. Add `compare-retrieval` to `.PHONY` and call the script.

Add `append_experiment_summary(evaluations, experiments_path, results_path)`. It writes the six real aggregate rows and chooses a strategy winner by the tuple `(recall_at_k, mrr, -mean_latency_ms)`, so the conclusion is derived from the stored measurements rather than manually entered.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/evals/test_retrieval_experiments.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add evals/compare_retrieval_experiments.py tests/evals/test_retrieval_experiments.py Makefile
git commit -m "eval: compare chunking hybrid retrieval and reranking"
```

### Task 5: Run experiments and record the measured decision

**Files:**
- Modify: `ROADMAP.md`

**Interfaces:**
- Consumes: the JSON path printed by `make compare-retrieval`.
- Produces: a real result JSON and an automatically appended dated table with the measured configuration decision.

- [ ] **Step 1: Run the controlled experiment**

Run: `make compare-retrieval`  
Expected: the command prints aggregate Recall@5, MRR, latency, and a JSON file under `evals/runs/`.

- [ ] **Step 2: Verify the recorded decision**

Run: `tail -n 20 EXPERIMENTS.md`  
Expected: a dated table for all six configurations, the exact JSON result path, and a decision generated from Recall@5, MRR, then latency.

- [ ] **Step 3: Mark roadmap complete**

Check every bullet in Roadmap block 6.5 only after the table contains real values and the result JSON exists.

- [ ] **Step 4: Run full verification**

Run: `make check`  
Expected: Ruff passes and all tests pass.

- [ ] **Step 5: Commit**

```bash
git add EXPERIMENTS.md ROADMAP.md evals/runs/retrieval-experiments-*.json
git commit -m "docs: record retrieval experiment results"
```
