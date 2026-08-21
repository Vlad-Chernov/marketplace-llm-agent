# Vector Policy Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Построить воспроизводимый ChromaDB-индекс политик и добавить векторный поиск с фильтрами.

**Architecture:** `vector.py` отделяет интерфейс эмбеддера от хранения ChromaDB, поэтому тесты используют детерминированный fake-эмбеддер без скачивания модели. Скрипт ingest загружает исходные Markdown-чанки и полностью пересоздаёт одну коллекцию `policy_chunks` в игнорируемом каталоге.

**Tech Stack:** Python 3.13, ChromaDB, sentence-transformers, Pydantic v2, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-20-vector-policy-retrieval-design.md`

## Global Constraints

- Модель: `intfloat/multilingual-e5-small`.
- Документы кодируются с `passage: `, запросы — с `query: `.
- Индекс находится в `data/vector_store/` и не хранится в Git.
- Коллекция Chroma называется `policy_chunks` и использует cosine distance.
- Поддерживаются только фильтры `document_id` и `policy_type`.
- Векторный поиск возвращает существующий `SearchResult` из `retrieval.lexical`.
- Сначала пишется тест, затем минимальная реализация.
- Проверка проекта выполняется командой `make check`.

---

## File Structure

- Modify: `pyproject.toml` and `uv.lock` — runtime dependencies.
- Modify: `.gitignore` — exclude `data/vector_store/`.
- Modify: `Makefile` — add `make ingest` and lint the new script.
- Create: `src/marketplace_agent/retrieval/vector.py` — embedder protocol, Chroma index and vector retriever.
- Create: `scripts/ingest_policies.py` — reproducible index rebuild.
- Create: `tests/retrieval/test_vector.py` — fake-embedder tests.

### Task 1: Add the vector retriever

**Files:**

- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Create: `src/marketplace_agent/retrieval/vector.py`
- Create: `tests/retrieval/test_vector.py`

**Interfaces:**

- Consumes: `PolicyChunk` and `SearchResult`.
- Produces: `SentenceTransformerEmbedder`, `build_vector_index`, `VectorRetriever.open` and `VectorRetriever.search`.
- Used later by: hybrid retrieval.

- [ ] **Step 1: Add dependencies**

```bash
uv add chromadb sentence-transformers
```

Expected: `pyproject.toml` and `uv.lock` contain the two runtime dependencies.

- [ ] **Step 2: Write failing tests**

```python
from pathlib import Path

from marketplace_agent.retrieval.documents import PolicyChunk
from marketplace_agent.retrieval.vector import (
    VectorRetriever,
    build_vector_index,
)


class FakeEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        normalized = text.casefold().replace("ё", "е")
        return [
            float(normalized.count("гарантия")),
            float(normalized.count("доставка")),
            float(normalized.count("возврат")),
        ]


def make_chunks() -> list[PolicyChunk]:
    return [
        PolicyChunk(
            chunk_id="returns-01",
            document_id="returns",
            policy_type="returns",
            heading="Возврат",
            text="Возврат товара доступен в течение 14 дней.",
        ),
        PolicyChunk(
            chunk_id="warranty-01",
            document_id="warranty",
            policy_type="warranty",
            heading="Гарантия",
            text="Гарантия покрывает производственные дефекты.",
        ),
        PolicyChunk(
            chunk_id="delivery-01",
            document_id="delivery",
            policy_type="delivery",
            heading="Доставка",
            text="Статус доставки отображается в заказе.",
        ),
    ]


def test_builds_reopens_and_searches_vector_index(tmp_path: Path) -> None:
    index_path = tmp_path / "vector_store"
    retriever = build_vector_index(make_chunks(), index_path, FakeEmbedder())

    results = retriever.search("гарантия", k=2)
    reopened = VectorRetriever.open(index_path, FakeEmbedder())

    assert results[0].chunk.chunk_id == "warranty-01"
    assert results[0].rank == 1
    assert results[0].score > 0
    assert reopened.search("доставка", k=1)[0].chunk.chunk_id == "delivery-01"


def test_filters_vector_results_before_ranking(tmp_path: Path) -> None:
    retriever = build_vector_index(
        make_chunks(),
        tmp_path / "vector_store",
        FakeEmbedder(),
    )

    results = retriever.search(
        "возврат",
        k=3,
        filters={"document_id": "returns"},
    )

    assert [result.chunk.chunk_id for result in results] == ["returns-01"]


def test_returns_empty_for_invalid_vector_requests(tmp_path: Path) -> None:
    retriever = build_vector_index(
        make_chunks(),
        tmp_path / "vector_store",
        FakeEmbedder(),
    )

    assert retriever.search("", k=1) == []
    assert retriever.search("гарантия", k=0) == []
    assert retriever.search(
        "гарантия",
        k=1,
        filters={"unknown": "value"},
    ) == []
```

- [ ] **Step 3: Run tests and confirm the expected failure**

```bash
uv run pytest tests/retrieval/test_vector.py -v
```

Expected: collection error because `marketplace_agent.retrieval.vector` does not exist.

- [ ] **Step 4: Implement the minimal vector retriever**

Define an `Embedder` protocol with these exact methods:

```python
class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_queries(self, texts: list[str]) -> list[list[float]]: ...
```

`SentenceTransformerEmbedder` loads `intfloat/multilingual-e5-small` lazily in its constructor. Its methods call `model.encode(..., normalize_embeddings=True)` with `passage: ` and `query: ` prefixes respectively, then return Python lists.

`build_vector_index` creates `chromadb.PersistentClient(path=str(path))`, removes the existing `policy_chunks` collection if present, then creates it with `metadata={"hnsw:space": "cosine"}`. Add each chunk with its embedding and metadata. Store the original text in metadata under `text`, because embeddings are supplied explicitly.

`VectorRetriever.search` rejects empty queries and non-positive `k`. It translates supported filters into Chroma `where`; unknown keys return `[]`. It queries with one query embedding, reconstructs `PolicyChunk` from the ID and metadata, converts cosine distance to `score = 1 - distance`, and returns one-based `SearchResult` ranks.

- [ ] **Step 5: Run focused tests**

```bash
uv run pytest tests/retrieval/test_vector.py -v
```

Expected: three passing tests.

### Task 2: Make index rebuilding reproducible

**Files:**

- Modify: `.gitignore`
- Modify: `Makefile`
- Create: `scripts/ingest_policies.py`

**Interfaces:**

- Consumes: `data/support/`, `load_policy_chunks`, `SentenceTransformerEmbedder` and `build_vector_index`.
- Produces: persistent collection under `data/vector_store/`.
- Used later by: `VectorRetriever.open` and the support-answer workflow.

- [ ] **Step 1: Write the ingestion script**

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marketplace_agent.retrieval.documents import load_policy_chunks
from marketplace_agent.retrieval.vector import (
    SentenceTransformerEmbedder,
    build_vector_index,
)


def main() -> None:
    chunks = load_policy_chunks(PROJECT_ROOT / "data" / "support")
    retriever = build_vector_index(
        chunks,
        PROJECT_ROOT / "data" / "vector_store",
        SentenceTransformerEmbedder(),
    )
    print(f"Indexed chunks: {retriever.count}")


if __name__ == "__main__":
    main()
```

Add `count` as a read-only `VectorRetriever` property returning the collection record count.

- [ ] **Step 2: Configure Git ignore and Makefile**

Append this exact line to `.gitignore`:

```text
data/vector_store/
```

Add to `Makefile`:

```makefile
ingest:
	uv run python scripts/ingest_policies.py
```

Add `scripts/ingest_policies.py` to both existing Ruff command lines in `Makefile`.

- [ ] **Step 3: Run focused tests and build the real index**

```bash
uv run pytest tests/retrieval/test_vector.py -v
make ingest
```

Expected: three tests pass; the first run downloads the embedding model, then prints `Indexed chunks: 12`.

- [ ] **Step 4: Run the project check**

```bash
make check
```

Expected: Ruff completes without errors and all tests pass.

- [ ] **Step 5: Commit**

```bash
git add .gitignore Makefile pyproject.toml uv.lock scripts/ingest_policies.py src/marketplace_agent/retrieval/vector.py tests/retrieval/test_vector.py
git commit -m "feat: add vector policy retrieval"
```

## Plan Self-Review

- Спецификация покрыта: выбранная модель, префиксы, cosine distance, Chroma persistence, фильтры, общий `SearchResult` и `make ingest`.
- Тесты не скачивают модель: они используют fake-эмбеддер и временный индекс.
- Не добавлены hybrid retrieval, reranking, LLM или генерация ответа.
