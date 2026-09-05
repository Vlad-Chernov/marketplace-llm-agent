# LangGraph Content Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Перенести цикл создания и исправления карточки в LangGraph, сохранив
`run_content_pipeline(product, llm, max_attempts)`.

**Architecture:** `pipeline.py` готовит товар, attributes, evidence и rules.
`content.graph` выполняет `generate → validate → repair` и возвращает
готовый `PipelineResult`.

**Tech Stack:** Python, LangGraph, Pydantic, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-05-langgraph-content-design.md`

## Global Constraints

- Не менять сигнатуру `run_content_pipeline`.
- Не менять модели домена и правила валидации.
- Максимум попыток: три по умолчанию.
- `manual_review` возвращается как `PipelineResult`.
- Тесты используют `FakeLLMClient`.
- Не добавлять LangChain.

---

### Task 1: Добавить graph карточки

**Files:**

- Create: `src/marketplace_agent/content/graph.py`
- Create: `tests/content/test_graph.py`

**Interfaces:**

```python
class ContentGraphState(TypedDict, total=False):
    product: Product
    extracted_attributes: AttributeExtractionResult
    evidence: GroundingEvidence
    deterministic_rules: list[dict[str, Any]]
    semantic_rules: list[dict[str, Any]]
    content: GeneratedContent | None
    violations: list[RuleViolation]
    attempts: int
    max_attempts: int
    seen_contents: set[str]
    result: PipelineResult


def build_content_graph(llm: LLMClient):
    """Return the compiled content graph."""
```

- [ ] **Step 1: Напиши падающий тест делегирования pipeline в graph**

В `tests/content/test_graph.py` добавь:

```python
from decimal import Decimal

import marketplace_agent.content.pipeline as pipeline

from marketplace_agent.catalog.extractor import (
    AttributeExtractionResult,
)
from marketplace_agent.domain.models import (
    PipelineResult,
    Product,
)
from marketplace_agent.llm.base import FakeLLMClient


def make_product() -> Product:
    return Product(
        sku="LAP-0001",
        category="laptops",
        brand="Lenovo",
        model="IdeaPad 1000",
        price=Decimal(75000),
        sales_count=10,
        supplier_description="Ноутбук Lenovo с экраном 14 дюймов.",
        attributes={},
    )


def test_pipeline_delegates_content_cycle_to_graph(
    monkeypatch,
) -> None:
    expected = PipelineResult(
        sku="LAP-0001",
        attempts=1,
        status="completed",
    )
    received_states: list[dict[str, object]] = []

    class Graph:
        def invoke(
            self,
            state: dict[str, object],
        ) -> dict[str, PipelineResult]:
            received_states.append(state)
            return {"result": expected}

    def build_graph(_: FakeLLMClient) -> Graph:
        return Graph()

    monkeypatch.setattr(
        pipeline,
        "extract_attributes",
        lambda *_: AttributeExtractionResult(
            attributes={},
            unsupported_facts=[],
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "build_content_graph",
        build_graph,
        raising=False,
    )

    result = pipeline.run_content_pipeline(
        make_product(),
        FakeLLMClient(),
    )

    assert result == expected
    assert received_states[0]["attempts"] == 0
    assert received_states[0]["max_attempts"] == 3
```

- [ ] **Step 2: Запусти тест**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/content/test_graph.py::test_pipeline_delegates_content_cycle_to_graph -v
```

Ожидается падение: текущий pipeline не вызывает graph.

- [ ] **Step 3: Создай `content.graph`**

В `src/marketplace_agent/content/graph.py`:

- создай `ContentGraphState`;
- добавь nodes `generate`, `validate`, `repair`, `completed`,
  `manual_review`;
- в `generate` вызови `generate_content` и увеличь `attempts`;
- в `repair` вызови `repair_content` и увеличь `attempts`;
- перенеси `_validate_content` и `_grounding_violations` из `pipeline.py`
  в graph-модуль;
- в `validate` вызови перенесённую validation-функцию и сохрани
  `violations`;
- создай fingerprint через `content.model_dump_json()`.

Маршрут после `validate`:

```text
нет violations → completed
fingerprint уже в seen_contents → manual_review
attempts равен max_attempts → manual_review
иначе → repair
```

Для первого невалидного контента добавляй fingerprint в `seen_contents`
до перехода в `repair`.

`completed` и `manual_review` создают `PipelineResult` из state.

- [ ] **Step 4: Запусти новый тест**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/content/test_graph.py::test_pipeline_delegates_content_cycle_to_graph -v
```

Тест всё ещё должен падать: `pipeline.py` пока использует старый цикл.

---

### Task 2: Перевести pipeline на graph

**Files:**

- Modify: `src/marketplace_agent/content/pipeline.py`
- Modify: `tests/content/test_pipeline.py`

- [ ] **Step 1: Замени цикл в `run_content_pipeline`**

Сохрани:

- проверку `max_attempts`;
- `load_attribute_specs`;
- `extract_attributes`;
- создание `GroundingEvidence`;
- `_load_rules`.

После подготовки данных выполни:

```python
graph = build_content_graph(llm)
result = graph.invoke(
    {
        "product": product,
        "extracted_attributes": extracted_attributes,
        "evidence": evidence,
        "deterministic_rules": deterministic_rules,
        "semantic_rules": semantic_rules,
        "content": None,
        "violations": [],
        "attempts": 0,
        "max_attempts": max_attempts,
        "seen_contents": set(),
    }
)

return result["result"]
```

Удали старый цикл `for attempt in range(1, max_attempts + 1)` и перенесённые
validation-helper функции.

- [ ] **Step 2: Запусти тест делегирования**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/content/test_graph.py::test_pipeline_delegates_content_cycle_to_graph -v
```

Ожидается `1 passed`.

- [ ] **Step 3: Запусти существующие сценарии pipeline**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/content/test_pipeline.py -v
```

Ожидаются четыре зелёных сценария: первая попытка, repair, повтор
контента и некорректный лимит.

- [ ] **Step 4: Добавь проверку трёх невалидных попыток**

В `tests/content/test_pipeline.py` создай `FakeLLMClient` с:

1. ответом extraction;
2. первой невалидной карточкой;
3. grounding без нарушений;
4. semantic с нарушением;
5. второй отличающейся невалидной карточкой;
6. grounding без нарушений;
7. semantic с нарушением;
8. третьей отличающейся невалидной карточкой;
9. grounding без нарушений;
10. semantic с нарушением.

Проверь:

```python
assert result.status == "manual_review"
assert result.attempts == 3
assert [item.rule_id for item in result.violations] == [
    "title-length"
]
```

- [ ] **Step 5: Проверь content graph**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/content/test_graph.py \
  tests/content/test_pipeline.py -v

UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run ruff check --fix \
  src/marketplace_agent/content/graph.py \
  src/marketplace_agent/content/pipeline.py \
  tests/content/test_graph.py \
  tests/content/test_pipeline.py

git diff --check
```

- [ ] **Step 6: Закоммить graph карточки**

```bash
git add src/marketplace_agent/content/graph.py \
  src/marketplace_agent/content/pipeline.py \
  tests/content/test_graph.py \
  tests/content/test_pipeline.py

git commit -m "feat: run content pipeline with LangGraph"
```

---

### Task 3: Завершить блок 12.3

**Files:**

- Modify: `ROADMAP.md`

- [ ] **Step 1: Запусти связанные тесты**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/content \
  tests/batch/test_processor.py \
  tests/evals/test_mvp_executor.py -v
```

- [ ] **Step 2: Проверь стиль**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run ruff check \
  src/marketplace_agent/content \
  tests/content \
  tests/batch/test_processor.py \
  tests/evals/test_mvp_executor.py

git diff --check
```

- [ ] **Step 3: Отметь блок**

В `ROADMAP.md` замени:

```md
- [ ] **12.3:** перенести цикл карточки с максимум тремя исправлениями и ручной очередью.
```

на:

```md
- [x] **12.3:** перенести цикл карточки с максимум тремя исправлениями и ручной очередью.
```

- [ ] **Step 4: Закоммить завершение блока**

```bash
git add ROADMAP.md
git commit -m "docs: complete LangGraph content milestone"

git status --short
git log --oneline -4
```