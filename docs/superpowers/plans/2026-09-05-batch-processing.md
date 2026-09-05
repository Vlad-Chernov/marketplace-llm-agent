# Batch Processing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Добавить безопасную конкурентную обработку карточек по SKU.

**Architecture:** `BatchProcessor` запускает существующий
`run_content_pipeline` в двух потоках. Каждый worker получает отдельный
LLM-клиент, а общий `RateLimiter` ограничивает все вызовы `chat`.

**Tech Stack:** Python, `concurrent.futures`, `threading`, Pydantic,
pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-05-batch-processing-design.md`

## Global Constraints

- Параллелизм по умолчанию: `2`.
- Лимит по умолчанию: `30` LLM-запросов в минуту.
- Повторные попытки, checkpoints и dead-letter queue не входят в 11.1.
- Тесты не обращаются к реальному API.
- Не изменять существующий `run_content_pipeline`.

---

### Task 1: Настройки и rate limiter

**Files:**

- Create: `src/marketplace_agent/batch/__init__.py`
- Create: `src/marketplace_agent/batch/limiter.py`
- Create: `tests/batch/test_limiter.py`
- Modify: `src/marketplace_agent/config.py`
- Modify: `tests/test_config.py`
- Modify: `.env.example`

**Interfaces:**

```python
class RateLimiter:
    def __init__(
        self,
        requests_per_minute: int,
        *,
        clock: Callable[[], float] = monotonic,
        sleeper: Callable[[float], None] = sleep,
    ) -> None: ...

    def acquire(self) -> None: ...


class RateLimitedLLMClient:
    def __init__(self, client: LLMClient, limiter: RateLimiter) -> None: ...

    def chat(...) -> LLMResponse: ...
    def embed(self, texts: list[str]) -> list[list[float]]: ...
```

- [ ] **Step 1: Напиши падающий тест limiter**

```python
def test_limiter_spaces_requests_by_configured_interval() -> None:
    now = 0.0
    sleeps: list[float] = []

    def clock() -> float:
        return now

    def sleeper(delay: float) -> None:
        nonlocal now
        sleeps.append(delay)
        now += delay

    limiter = RateLimiter(30, clock=clock, sleeper=sleeper)
    limiter.acquire()
    limiter.acquire()
    limiter.acquire()

    assert sleeps == [2.0, 2.0]
```

- [ ] **Step 2: Запусти тест**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/batch/test_limiter.py::test_limiter_spaces_requests_by_configured_interval -v
```

Ожидается `ModuleNotFoundError`.

- [ ] **Step 3: Реализуй limiter**

`RateLimiter` хранит `_next_allowed_at` и `Lock`. Внутри `acquire()`:
под lock вычисляет ожидание, сдвигает `_next_allowed_at` на
`60 / requests_per_minute`, затем выполняет `sleeper(delay)` уже без lock.
Если `requests_per_minute < 1`, выбрасывает `ValueError`.

`RateLimitedLLMClient.chat()` вызывает `limiter.acquire()`, затем
делегирует запрос исходному клиенту. `embed()` просто делегируется.

- [ ] **Step 4: Добавь настройки batch**

В `Settings` добавь поля:

```python
batch_max_workers: int = 2
batch_llm_requests_per_minute: int = 30
```

В `from_environment()` загрузи:

```python
batch_max_workers=int(os.getenv("BATCH_MAX_WORKERS", "2"))
batch_llm_requests_per_minute=int(
    os.getenv("BATCH_LLM_REQUESTS_PER_MINUTE", "30")
)
```

Перед `return cls(...)` проверь оба значения. При значении `< 1`
выбрасывай `ValueError` с именем соответствующей переменной.

В `.env.example` добавь:

```dotenv
# Пакетная обработка карточек.
BATCH_MAX_WORKERS=2
BATCH_LLM_REQUESTS_PER_MINUTE=30
```

- [ ] **Step 5: Добавь тест настроек**

```python
def test_loads_batch_settings(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
    monkeypatch.setenv("BATCH_MAX_WORKERS", "2")
    monkeypatch.setenv("BATCH_LLM_REQUESTS_PER_MINUTE", "30")

    settings = Settings.from_environment()

    assert settings.batch_max_workers == 2
    assert settings.batch_llm_requests_per_minute == 30
```

- [ ] **Step 6: Проверь и закоммить**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/batch/test_limiter.py tests/test_config.py -v

UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run ruff check --fix \
  src/marketplace_agent/batch/limiter.py \
  tests/batch/test_limiter.py \
  src/marketplace_agent/config.py \
  tests/test_config.py

git diff --check
git add .env.example src/marketplace_agent/batch \
  src/marketplace_agent/config.py tests/batch/test_limiter.py \
  tests/test_config.py
git commit -m "feat: add batch rate limiter"
```

### Task 2: BatchProcessor

**Files:**

- Create: `src/marketplace_agent/batch/processor.py`
- Create: `tests/batch/test_processor.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class BatchSummary:
    requested_skus: list[str]
    results: dict[str, PipelineResult]
    errors: dict[str, str]
    elapsed_seconds: float


class BatchProcessor:
    def __init__(
        self,
        products: ProductRepository,
        llm_factory: Callable[[], LLMClient],
        *,
        max_workers: int = 2,
        requests_per_minute: int = 30,
        pipeline: Callable[[Product, LLMClient], PipelineResult] =
            run_content_pipeline,
    ) -> None: ...

    def run(
        self,
        product_ids: list[str],
        resume: bool = True,
    ) -> BatchSummary: ...
```

- [ ] **Step 1: Напиши падающий тест на продолжение после ошибки**

```python
def test_batch_continues_after_one_product_fails() -> None:
    processor = BatchProcessor(
        products=FakeProducts({"LAP-001": product_one, "LAP-002": product_two}),
        llm_factory=FakeLLMClient,
        pipeline=fail_for_first_product,
    )

    summary = processor.run(["LAP-001", "LAP-002"])

    assert list(summary.results) == ["LAP-002"]
    assert "LAP-001" in summary.errors
```

- [ ] **Step 2: Напиши падающий тест на два worker**

Тестовая функция увеличивает общий счётчик активных задач, ждёт через
`threading.Barrier(2)`, затем уменьшает счётчик. Проверка:

```python
assert maximum_active == 2
```

Вызов:

```python
summary = processor.run(["LAP-001", "LAP-002", "LAP-003"])
assert len(summary.results) == 3
```

- [ ] **Step 3: Реализуй BatchProcessor**

- Отклони пустой или повторяющийся SKU через `ValueError`.
- Создай один `RateLimiter` на весь вызов `run`.
- Используй `ThreadPoolExecutor(max_workers=self.max_workers)`.
- В worker получи товар через `products.get_by_sku(sku)`.
- Если товара нет, верни ошибку `Product not found.`.
- Создай клиент через `llm_factory()`, оберни в
  `RateLimitedLLMClient`, передай в `pipeline`.
- Собери futures, но заполни `results` и `errors` проходом по исходному
  `product_ids`; это сохраняет порядок.
- `resume` в 11.1 не влияет на выполнение.
- Замерь `elapsed_seconds` через `monotonic()`.

- [ ] **Step 4: Добавь тесты границ**

```python
def test_batch_rejects_duplicate_skus() -> None:
    with pytest.raises(ValueError, match="unique"):
        processor.run(["LAP-001", "LAP-001"])


def test_batch_records_unknown_sku_and_processes_known_one() -> None:
    summary = processor.run(["MISSING-001", "LAP-001"])

    assert list(summary.results) == ["LAP-001"]
    assert summary.errors == {"MISSING-001": "Product not found."}
```

- [ ] **Step 5: Проверь и закоммить**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/batch/test_processor.py -v

UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run ruff check --fix \
  src/marketplace_agent/batch/processor.py \
  tests/batch/test_processor.py

git diff --check
git add src/marketplace_agent/batch/processor.py \
  tests/batch/test_processor.py
git commit -m "feat: process product cards concurrently"
```

### Task 3: Общая проверка 11.1

**Files:**

- Modify: `ROADMAP.md`

- [ ] **Step 1: Запусти все связанные тесты**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/batch \
  tests/content/test_pipeline.py \
  tests/test_config.py -v
```

- [ ] **Step 2: Проверь стиль и изменения**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run ruff check \
  src/marketplace_agent/batch \
  src/marketplace_agent/config.py \
  tests/batch \
  tests/test_config.py

git diff --check
```

- [ ] **Step 3: Отметь этап**

В `ROADMAP.md` измени только:

```md
- [x] **11.1:** добавить ограниченный пул конкурентных задач и rate limiter.
```

- [ ] **Step 4: Зафиксируй результат**

```bash
git add ROADMAP.md
git commit -m "docs: complete batch concurrency milestone"

git status --short
git log --oneline -3
```