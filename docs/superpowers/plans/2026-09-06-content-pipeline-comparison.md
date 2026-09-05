# Content Pipeline Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` task-by-task. Steps use checkboxes.

**Goal:** Сравнить прежний линейный content pipeline и LangGraph-граф на
одинаковом наборе из 12 товаров с реальным LLM.

**Architecture:** Прежний цикл хранится только в
`src/marketplace_agent/evals/legacy_content_pipeline.py`.
Общий evaluation-исполнитель запускает обе реализации, измеряет метрики и
сохраняет технический trace по каждому SKU. Production-конвейер не меняется.

**Tech Stack:** Python, LangGraph, pytest, Ruff, JSON, реальный LLM-провайдер.

**Spec:** `docs/superpowers/specs/2026-09-05-content-pipeline-comparison-design.md`

## Global Constraints

- Не менять `run_content_pipeline(...)` и production-код графа.
- Legacy существует только для evaluation.
- Использовать 12 уникальных SKU из фиксированного manifest.
- Обе версии используют `max_attempts=3`.
- Live-команда всегда запускается с `LLM_PROGRESS=1`.
- Не сохранять prompts, ответы модели, ключи или PII в trace.
- Не добавлять неотслеживаемые личные файлы в git.

---

### Task 1: Расширить модель результатов и метрики

**Files:**

- Modify: `src/marketplace_agent/evals/content_pipeline.py`
- Modify: `tests/evals/test_content_pipeline_comparison.py`

**Interfaces:**

```python
Pipeline = Callable[[Product, LLMClient, int], PipelineResult]

@dataclass(frozen=True)
class ContentPipelineCaseResult:
    sku: str
    status: str
    attempts: int
    true_attributes: Mapping[str, Any]
    extracted_attributes: Mapping[str, Any]
    used_attributes: Mapping[str, Any]
    violations: list[RuleViolation]
    latency_ms: int
    cost_usd: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str | None = None
    error: str | None = None
    trace: list[dict[str, object]] = field(default_factory=list)

@dataclass(frozen=True)
class PipelineVersionMetrics:
    success_rate: float
    attribute_f1: float
    hallucination_rate: float
    violation_count: int
    failures_by_type: dict[str, int]
    average_latency_ms: float
    total_cost_usd: float
    prompt_tokens: int
    completion_tokens: int
```

- [ ] **Step 1: Добавь падающий тест метрик статусов**

В `tests/evals/test_content_pipeline_comparison.py` добавь кейсы:

```python
results = [
    ContentPipelineCaseResult(
        sku="LAP-0001",
        status="completed",
        attempts=1,
        true_attributes={},
        extracted_attributes={},
        used_attributes={},
        violations=[],
        latency_ms=100,
        cost_usd=0.01,
    ),
    ContentPipelineCaseResult(
        sku="LAP-0002",
        status="manual_review",
        attempts=3,
        true_attributes={},
        extracted_attributes={},
        used_attributes={},
        violations=[violation("title-length")],
        latency_ms=200,
        cost_usd=0.02,
    ),
    ContentPipelineCaseResult(
        sku="LAP-0003",
        status="error",
        attempts=0,
        true_attributes={},
        extracted_attributes={},
        used_attributes={},
        violations=[],
        latency_ms=300,
        cost_usd=0.03,
        error="LLMProviderError: HTTP 429",
    ),
]
```

Проверь:

```python
metrics = _calculate_version_metrics(results)

assert metrics.success_rate == pytest.approx(1 / 3)
assert metrics.failures_by_type == {
    "LLMProviderError": 1,
    "manual_review": 1,
}
assert metrics.prompt_tokens == 0
assert metrics.completion_tokens == 0
```

- [ ] **Step 2: Запусти тест и зафиксируй падение**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/evals/test_content_pipeline_comparison.py -v
```

Ожидается ошибка: у метрик нет `success_rate` или `failures_by_type`.

- [ ] **Step 3: Добавь поля и расчёт**

- Добавь `status`, `attempts` и `trace` в case result.
- Добавь метрики из интерфейса выше.
- `completed` — единственный успешный статус.
- Для `manual_review` увеличивай `failures_by_type["manual_review"]`.
- Для `error` используй часть строки до первого `:`, например
  `LLMProviderError`.
- Суммируй токены по всем cases.
- Сохрани существующие F1, hallucination rate, violations, latency и cost.

- [ ] **Step 4: Запусти тесты**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/evals/test_content_pipeline_comparison.py -v
```

Ожидается: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/marketplace_agent/evals/content_pipeline.py \
  tests/evals/test_content_pipeline_comparison.py
git commit -m "feat: measure content pipeline outcomes"
```

---

### Task 2: Добавить legacy-конвейер только для evaluation

**Files:**

- Create: `src/marketplace_agent/evals/legacy_content_pipeline.py`
- Create: `tests/evals/test_legacy_content_pipeline.py`

**Interfaces:**

```python
def run_legacy_content_pipeline(
    product: Product,
    llm: LLMClient,
    max_attempts: int = 3,
) -> PipelineResult:
    """Run the pre-LangGraph linear content cycle for evaluation only."""
```

- [ ] **Step 1: Добавь падающий тест поведения**

Скопируй сценарий успешной первой попытки из
`tests/content/test_pipeline.py`, но импортируй:

```python
from marketplace_agent.evals.legacy_content_pipeline import (
    run_legacy_content_pipeline,
)
```

Проверь:

```python
result = run_legacy_content_pipeline(make_product(), client)

assert result.status == "completed"
assert result.attempts == 1
```

- [ ] **Step 2: Запусти тест**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/evals/test_legacy_content_pipeline.py -v
```

Ожидается `ModuleNotFoundError`.

- [ ] **Step 3: Создай legacy implementation**

В `src/marketplace_agent/evals/legacy_content_pipeline.py` перенеси
реализацию из:

```bash
git show d50decb^:src/marketplace_agent/content/pipeline.py
```

Сохрани линейный цикл, включая:

- проверку `max_attempts`;
- extraction и создание `GroundingEvidence`;
- fingerprint через `content.model_dump_json()`;
- повтор fingerprint → `manual_review`;
- лимит попыток → `manual_review`;
- `_validate_content` и `_grounding_violations`.

Не импортируй `run_content_pipeline` из production-пакета.

- [ ] **Step 4: Добавь тест отсутствия production-импорта**

```python
def test_legacy_pipeline_is_not_imported_by_production() -> None:
    source = Path(
        "src/marketplace_agent/content/pipeline.py"
    ).read_text(encoding="utf-8")

    assert "legacy_content_pipeline" not in source
```

- [ ] **Step 5: Запусти тесты**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/evals/test_legacy_content_pipeline.py \
  tests/content/test_pipeline.py -v
```

Ожидается: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/marketplace_agent/evals/legacy_content_pipeline.py \
  tests/evals/test_legacy_content_pipeline.py
git commit -m "feat: preserve legacy pipeline for evaluation"
```

---

### Task 3: Запуск обеих версий и JSON traces

**Files:**

- Modify: `src/marketplace_agent/evals/content_pipeline.py`
- Modify: `evals/compare_content_pipeline.py`
- Modify: `tests/evals/test_content_pipeline_comparison.py`
- Modify: `data/gold/content_pipeline_manifest.json`

**Interfaces:**

```python
def run_pipeline_version(
    products: Sequence[Product],
    llm_factory: Callable[[], LLMClient],
    pipeline: Pipeline,
    max_attempts: int,
    input_price_per_million: float,
    output_price_per_million: float,
) -> list[ContentPipelineCaseResult]:
    """Run one named implementation against fixed products."""
```

- [ ] **Step 1: Добавь падающий тест trace и раздельных вызовов**

Создай `CountingFactory`, который увеличивает счётчик при создании клиента.
Передай два pipeline-stub: один возвращает `completed`, другой —
`manual_review`.

Проверь:

```python
assert len(legacy_results) == 2
assert len(graph_results) == 2
assert factory.calls == 4
assert legacy_results[0].trace[-1]["event_type"] == "completed"
assert graph_results[0].trace[-1]["event_type"] == "manual_review"
```

- [ ] **Step 2: Запусти тест**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/evals/test_content_pipeline_comparison.py -v
```

Ожидается падение: старый `run_pipeline_version` принимает готовый клиент и
не формирует trace.

- [ ] **Step 3: Измени executor**

- Принимай фабрику клиента и pipeline-функцию.
- Создавай новый клиент для каждого SKU.
- Добавляй в trace событие `started`.
- При результате добавляй `completed` либо `manual_review` с `attempts` и
  списком `rule_id`.
- При исключении добавляй `error` с именем исключения и безопасным текстом.
- Не добавляй в trace supplier description, prompts или content.
- Для graph передавай `run_content_pipeline`.
- Для legacy передавай `run_legacy_content_pipeline`.

- [ ] **Step 4: Расширь manifest**

Замени `skus` на:

```json
[
  "LAP-0001", "LAP-0002", "LAP-0003", "LAP-0004",
  "LAP-0005", "LAP-0006", "LAP-0007", "LAP-0008",
  "LAP-0009", "LAP-0010", "LAP-0011", "LAP-0012"
]
```

- [ ] **Step 5: Обнови CLI**

В `evals/compare_content_pipeline.py`:

- замени `--version` на обязательные `--legacy-version` и
  `--graph-version`;
- цены берутся из `Settings.input_price_per_million` и
  `Settings.output_price_per_million`, если CLI-аргументы не переданы;
- создай две версии через новую сигнатуру executor;
- сохрани `created_at`, `manifest_sha256`, provider, модель, цены,
  `max_attempts`, results и comparison;
- не используй `CachedLLMClient`;
- напечатай success rate, latency, стоимость и failures by type.

- [ ] **Step 6: Проверь JSON и тесты**

```bash
PYTHONPATH=src UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache \
  uv run python -m json.tool data/gold/content_pipeline_manifest.json > /dev/null

UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/evals/test_content_pipeline_comparison.py \
  tests/evals/test_legacy_content_pipeline.py -v
```

Ожидается: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/marketplace_agent/evals/content_pipeline.py \
  evals/compare_content_pipeline.py \
  data/gold/content_pipeline_manifest.json \
  tests/evals/test_content_pipeline_comparison.py \
  tests/evals/test_legacy_content_pipeline.py
git commit -m "feat: compare legacy and graph content pipelines"
```

---

### Task 4: Выполнить live-эксперимент и оформить выводы

**Files:**

- Modify: `EXPERIMENTS.md`
- Modify: `docs/AGENT_FAILURES.md`
- Create: новый JSON run в `evals/runs/`, имя печатает CLI.

- [ ] **Step 1: Запусти полный локальный набор**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache make check
git diff --check
```

Ожидается: все локальные проверки проходят.

- [ ] **Step 2: Выполни live-запуск**

Подставь реальные цены из `.env`:

```bash
LLM_PROGRESS=1 \
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache \
uv run python evals/compare_content_pipeline.py \
  --legacy-version legacy-linear-v1 \
  --graph-version langgraph-v1 \
  --max-attempts 3
```

Сохрани путь к созданному JSON.

- [ ] **Step 3: Добавь раздел в `EXPERIMENTS.md`**

Укажи:

- дату и `run_id`;
- provider, модель, цены и manifest hash;
- таблицу legacy vs graph: success rate, F1, violations, latency, tokens,
  cost и failures by type;
- ссылку на JSON;
- вывод, есть ли измеримая польза LangGraph.

- [ ] **Step 4: Добавь восемь failure-разборов**

Для каждого неуспеха добавь отдельный разбор. Заголовок содержит реальные
`run_id`, версию и SKU из JSON run.

В каждом разборе укажи:

- Scenario;
- Expected behaviour;
- Actual behaviour;
- Trace summary;
- Root cause;
- Fix or mitigation.

Если в одном JSON меньше восьми failures, повтори live-запуск на том же
manifest и добавь случаи из нового JSON.

- [ ] **Step 5: Финальная проверка и commit**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache make check
git diff --check

git add EXPERIMENTS.md \
  docs/AGENT_FAILURES.md \
  evals/runs/content-pipeline-*.json
git commit -m "evals: document LangGraph pipeline comparison"
git status --short
```

---

### Task 5: Закрыть блок 12.4

**Files:**

- Modify: `ROADMAP.md`

- [ ] **Step 1: Отметь 12.4**

Замени:

```md
- [ ] **12.4:**
```

на:

```md
- [x] **12.4:**
```

Только после наличия JSON, таблицы эксперимента и восьми failure-разборов.

- [ ] **Step 2: Commit**

```bash
git add ROADMAP.md
git commit -m "docs: complete content pipeline comparison milestone"
git status --short
```
