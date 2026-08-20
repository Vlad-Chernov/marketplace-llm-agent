# BM25 Policy Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Добавить детерминированный BM25-поиск по чанкам политик поддержки.

**Architecture:** `BM25Retriever` получает готовые `PolicyChunk`, строит индекс в памяти и возвращает `SearchResult`. Нормализация запроса и текста одинакова; точные фильтры применяются до ранжирования. Внешние библиотеки для поиска не используются.

**Tech Stack:** Python 3.13, Pydantic v2, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-20-bm25-policy-retrieval-design.md`

## Global Constraints

- Рабочий корпус создаётся функцией `load_policy_chunks(path)`.
- Поддерживаются только фильтры `document_id` и `policy_type`.
- Нормализация: `casefold`, `ё → е`, русские, латинские и числовые токены.
- Параметры BM25: `k1 = 1.5`, `b = 0.75`.
- Поиск, эмбеддинги, LLM-вызовы и гибридное ранжирование не входят в блок.
- Сначала пишется тест, затем минимальная реализация.
- Проверка проекта выполняется командой `make check`.

---

## File Structure

- Modify: `src/marketplace_agent/retrieval/lexical.py` — BM25, нормализация и результаты поиска.
- Create: `tests/retrieval/test_lexical.py` — поведение BM25 на реальном и минимальном корпусе.

### Task 1: Add lexical BM25 retrieval

**Files:**

- Create: `src/marketplace_agent/retrieval/lexical.py`
- Create: `tests/retrieval/test_lexical.py`

**Interfaces:**

- Consumes: `PolicyChunk` from `marketplace_agent.retrieval.documents`.
- Produces: `SearchResult` and `BM25Retriever.search(query, k, filters=None)`.
- Used later by: hybrid retrieval and the policy-answer pipeline.

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path

from marketplace_agent.retrieval.documents import (
    PolicyChunk,
    load_policy_chunks,
)
from marketplace_agent.retrieval.lexical import BM25Retriever


def test_finds_exact_policy_terms_in_expected_chunk() -> None:
    chunks = load_policy_chunks(Path("data/support"))
    retriever = BM25Retriever(chunks)

    results = retriever.search("гарантия клавиатуры", k=2)

    assert results[0].chunk.chunk_id == "warranty-02"
    assert results[0].rank == 1
    assert results[0].score > 0


def test_filters_results_by_document_id_before_ranking() -> None:
    chunks = load_policy_chunks(Path("data/support"))
    retriever = BM25Retriever(chunks)

    results = retriever.search(
        "товар",
        k=5,
        filters={"document_id": "returns"},
    )

    assert results
    assert {result.chunk.document_id for result in results} == {"returns"}


def test_normalizes_yo_character_for_search() -> None:
    chunk = PolicyChunk(
        chunk_id="sample-01",
        document_id="sample",
        policy_type="sample",
        heading="Тест",
        text="Ёмкость аккумулятора указана в характеристиках.",
    )
    retriever = BM25Retriever([chunk])

    results = retriever.search("емкость", k=1)

    assert [result.chunk.chunk_id for result in results] == ["sample-01"]


def test_assigns_ranks_and_returns_empty_for_invalid_requests() -> None:
    chunks = load_policy_chunks(Path("data/support"))
    retriever = BM25Retriever(chunks)

    results = retriever.search("гарантия", k=5)

    assert [result.rank for result in results] == list(
        range(1, len(results) + 1)
    )
    assert results == sorted(
        results,
        key=lambda result: (-result.score, result.chunk.chunk_id),
    )
    assert retriever.search("", k=5) == []
    assert retriever.search("гарантия", k=0) == []
    assert retriever.search(
        "гарантия",
        k=5,
        filters={"unknown": "value"},
    ) == []
```

- [ ] **Step 2: Run tests and confirm the expected failure**

```bash
uv run pytest tests/retrieval/test_lexical.py -v
```

Expected: collection error because `marketplace_agent.retrieval.lexical` does not exist.

- [ ] **Step 3: Implement the minimal BM25 retriever**

Create `src/marketplace_agent/retrieval/lexical.py` with:

```python
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from marketplace_agent.retrieval.documents import PolicyChunk

K1 = 1.5
B = 0.75
TOKEN_PATTERN = re.compile(r"[a-zа-я0-9]+")


@dataclass(frozen=True)
class SearchResult:
    """Store one ranked policy chunk."""

    chunk: PolicyChunk
    score: float
    rank: int


class BM25Retriever:
    """Search policy chunks using an in-memory BM25 index."""

    def __init__(self, chunks: Sequence[PolicyChunk]) -> None:
        self._chunks = list(chunks)
        self._term_frequencies = [
            Counter(_normalize(chunk.text)) for chunk in self._chunks
        ]
        self._document_lengths = [
            sum(frequencies.values())
            for frequencies in self._term_frequencies
        ]
        self._average_document_length = (
            sum(self._document_lengths) / len(self._document_lengths)
            if self._document_lengths
            else 0.0
        )
        self._document_frequencies = Counter(
            term
            for frequencies in self._term_frequencies
            for term in frequencies
        )

    def search(
        self,
        query: str,
        k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[SearchResult]:
        """Return the highest-scoring chunks for a query."""

        query_terms = _normalize(query)
        if not query_terms or k <= 0:
            return []

        scored_chunks = [
            (chunk, self._score(index, query_terms))
            for index, chunk in enumerate(self._chunks)
            if _matches_filters(chunk, filters)
        ]
        ranked_chunks = sorted(
            (
                (chunk, score)
                for chunk, score in scored_chunks
                if score > 0
            ),
            key=lambda item: (-item[1], item[0].chunk_id),
        )
        return [
            SearchResult(chunk=chunk, score=score, rank=rank)
            for rank, (chunk, score) in enumerate(ranked_chunks[:k], start=1)
        ]

    def _score(self, index: int, query_terms: list[str]) -> float:
        if self._average_document_length == 0:
            return 0.0

        frequencies = self._term_frequencies[index]
        document_length = self._document_lengths[index]
        score = 0.0

        for term in set(query_terms):
            term_frequency = frequencies[term]
            if term_frequency == 0:
                continue
            document_frequency = self._document_frequencies[term]
            inverse_document_frequency = math.log(
                1
                + (
                    len(self._chunks) - document_frequency + 0.5
                )
                / (document_frequency + 0.5)
            )
            denominator = term_frequency + K1 * (
                1 - B + B * document_length / self._average_document_length
            )
            score += inverse_document_frequency * (
                term_frequency * (K1 + 1) / denominator
            )

        return score


def _normalize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold().replace("ё", "е"))


def _matches_filters(
    chunk: PolicyChunk,
    filters: Mapping[str, str] | None,
) -> bool:
    if filters is None:
        return True

    return all(
        key in {"document_id", "policy_type"}
        and getattr(chunk, key) == value
        for key, value in filters.items()
    )
```

- [ ] **Step 4: Run focused tests**

```bash
uv run pytest tests/retrieval/test_lexical.py -v
```

Expected: four passing tests.

- [ ] **Step 5: Run the project check**

```bash
make check
```

Expected: Ruff completes without errors and all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/marketplace_agent/retrieval/lexical.py tests/retrieval/test_lexical.py
git commit -m "feat: add BM25 policy retrieval"
```

## Plan Self-Review

- Спецификация покрыта: BM25, точные фильтры, нормализация, ранги, оценки и пустые результаты.
- Не добавлены внешние зависимости, эмбеддинги, LLM или гибридное ранжирование.
- Все используемые типы и интерфейсы определены в этом плане или в предыдущем блоке 6.1.
