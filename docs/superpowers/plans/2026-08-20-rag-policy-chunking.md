# RAG Policy Chunking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Разобрать Markdown-политики поддержки в цитируемые чанки с устойчивыми идентификаторами.

**Architecture:** Модуль `retrieval.documents` читает Markdown-файлы из одной папки и возвращает Pydantic-модели `PolicyChunk`. Документы с `##` режутся по разделам; документы без `##` — по непустым абзацам после `#`. Текст чанков сохраняется без перефразирования.

**Tech Stack:** Python 3.13, Pydantic v2, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-20-rag-policy-chunking-design.md`

## Global Constraints

- Рабочие документы находятся в `data/support/`.
- Чанк обязан иметь устойчивый ID, источник, заголовок и непустой исходный текст.
- Поиск, эмбеддинги, генерация ответа и проверка цитат не входят в этот блок.
- Сначала пишется тест, затем минимальная реализация.
- Проверка проекта выполняется командой `make check`.

---

## File Structure

- Create: `src/marketplace_agent/retrieval/__init__.py` — пакет retrieval.
- Create: `src/marketplace_agent/retrieval/documents.py` — модель чанка и Markdown-парсер.
- Create: `tests/retrieval/test_documents.py` — тесты реального корпуса и граничных случаев.

### Task 1: Parse support policies into chunks

**Files:**

- Create: `src/marketplace_agent/retrieval/__init__.py`
- Create: `src/marketplace_agent/retrieval/documents.py`
- Create: `tests/retrieval/test_documents.py`

**Interfaces:**

- Consumes: Markdown-файлы из `data/support/`.
- Produces: `PolicyChunk` и `load_policy_chunks(path) -> list[PolicyChunk]`.
- Used later by: `BM25Retriever`, `VectorRetriever` и генератор ответа с цитатами.

- [ ] **Step 1: Write failing tests**

```python
from pathlib import Path

import pytest

from marketplace_agent.retrieval.documents import load_policy_chunks

SUPPORT_PATH = Path("data/support")


def test_loads_citable_chunks_from_support_policies() -> None:
    chunks = load_policy_chunks(SUPPORT_PATH)

    assert len(chunks) == 12
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert {chunk.document_id for chunk in chunks} == {
        "delivery",
        "exchange",
        "returns",
        "warranty",
    }
    assert all(chunk.heading for chunk in chunks)
    assert all(chunk.text for chunk in chunks)

    returns_chunks = [
        chunk for chunk in chunks if chunk.document_id == "returns"
    ]
    assert [chunk.chunk_id for chunk in returns_chunks] == [
        "returns-01",
        "returns-02",
        "returns-03",
    ]
    assert returns_chunks[0].heading == "Возврат товара"
    assert "14 календарных дней" in returns_chunks[0].text


def test_splits_document_with_second_level_headings(tmp_path: Path) -> None:
    policy_path = tmp_path / "sample.md"
    policy_path.write_text(
        "# Тестовая политика\n\n"
        "## Первый пункт\n\n"
        "Первое правило.\n\n"
        "## Второй пункт\n\n"
        "Второе правило.\n",
        encoding="utf-8",
    )

    chunks = load_policy_chunks(tmp_path)

    assert [(chunk.chunk_id, chunk.heading, chunk.text) for chunk in chunks] == [
        ("sample-01", "Первый пункт", "Первое правило."),
        ("sample-02", "Второй пункт", "Второе правило."),
    ]


def test_rejects_policy_without_non_empty_text(tmp_path: Path) -> None:
    (tmp_path / "empty.md").write_text(
        "# Пустая политика\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="no non-empty policy text"):
        load_policy_chunks(tmp_path)
```

- [ ] **Step 2: Run tests and confirm the expected failure**

```bash
uv run pytest tests/retrieval/test_documents.py -v
```

Expected: collection error because `marketplace_agent.retrieval` does not exist.

- [ ] **Step 3: Create the package and minimal parser**

Create an empty `src/marketplace_agent/retrieval/__init__.py`.

Create `src/marketplace_agent/retrieval/documents.py` with:

```python
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class PolicyChunk(BaseModel):
    """Store one source fragment for retrieval and citation."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    policy_type: str = Field(min_length=1)
    heading: str = Field(min_length=1)
    text: str = Field(min_length=1)


def load_policy_chunks(path: Path | str) -> list[PolicyChunk]:
    """Load support Markdown documents as structured citation chunks."""

    directory = Path(path)
    chunks: list[PolicyChunk] = []
    document_ids: set[str] = set()

    for document_path in sorted(directory.glob("*.md")):
        document_id = document_path.stem
        if document_id in document_ids:
            raise ValueError(f"Duplicate document ID: {document_id}")
        document_ids.add(document_id)
        chunks.extend(
            _parse_document(
                document_path.read_text(encoding="utf-8"),
                document_id,
            )
        )

    if not chunks:
        raise ValueError("No policy chunks found.")

    return chunks


def _parse_document(content: str, document_id: str) -> list[PolicyChunk]:
    lines = content.splitlines()
    document_heading = _read_document_heading(lines)
    sections = _split_by_second_level_headings(lines)

    if sections:
        parts = sections
    else:
        parts = [
            (document_heading, paragraph)
            for paragraph in _read_paragraphs_after_heading(lines)
        ]

    if not parts:
        raise ValueError(f"{document_id}: no non-empty policy text")

    return [
        PolicyChunk(
            chunk_id=f"{document_id}-{index:02d}",
            document_id=document_id,
            policy_type=document_id,
            heading=heading,
            text=text,
        )
        for index, (heading, text) in enumerate(parts, start=1)
    ]


def _read_document_heading(lines: list[str]) -> str:
    for line in lines:
        if line.startswith("# "):
            return line.removeprefix("# ").strip()
    raise ValueError("Document heading is required.")


def _split_by_second_level_headings(
    lines: list[str],
) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    heading: str | None = None
    text_lines: list[str] = []

    for line in lines:
        if line.startswith("## "):
            if heading is not None:
                text = "\n".join(text_lines).strip()
                if not text:
                    raise ValueError(f"{heading}: no non-empty policy text")
                sections.append((heading, text))
            heading = line.removeprefix("## ").strip()
            text_lines = []
        elif heading is not None:
            text_lines.append(line)

    if heading is not None:
        text = "\n".join(text_lines).strip()
        if not text:
            raise ValueError(f"{heading}: no non-empty policy text")
        sections.append((heading, text))

    return sections


def _read_paragraphs_after_heading(lines: list[str]) -> list[str]:
    paragraphs: list[str] = []
    paragraph_lines: list[str] = []
    after_heading = False

    for line in lines:
        if line.startswith("# "):
            after_heading = True
            continue
        if not after_heading:
            continue
        if line.strip():
            paragraph_lines.append(line.strip())
        elif paragraph_lines:
            paragraphs.append(" ".join(paragraph_lines))
            paragraph_lines = []

    if paragraph_lines:
        paragraphs.append(" ".join(paragraph_lines))

    return paragraphs
```

- [ ] **Step 4: Run focused tests**

```bash
uv run pytest tests/retrieval/test_documents.py -v
```

Expected: three passing tests.

- [ ] **Step 5: Run the project check**

```bash
make check
```

Expected: Ruff completes without errors and all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/marketplace_agent/retrieval/__init__.py src/marketplace_agent/retrieval/documents.py tests/retrieval/test_documents.py
git commit -m "feat: parse support policies into chunks"
```

## Plan Self-Review

- Спецификация покрыта: модель, структурная нарезка, fallback для абзацев, устойчивые ID, исходный текст и тесты цитируемости.
- Поиск и генерация ответа остаются в следующих блоках.
- Интерфейс `load_policy_chunks(path) -> list[PolicyChunk]` одинаков во всех задачах.
