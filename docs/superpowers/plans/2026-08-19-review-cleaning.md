# Review Cleaning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Детерминированно очистить отзывы, замаскировать ПДн и удалить мусор и дубликаты.

**Architecture:** Новый модуль `reviews.cleaning` принимает список `Review` и возвращает очищенные отзывы вместе с ID обработанных записей. Обработка идёт в порядке входа: нормализация и маскирование ПДн, фильтрация мусора, дедупликация внутри SKU.

**Tech Stack:** Python 3.13, Pydantic v2, стандартные `html`, `re`, `difflib`, pytest.

**Spec:** `docs/superpowers/specs/2026-08-19-review-cleaning-design.md`

## Global Constraints

- LLM и эмбеддинги не используются.
- Сравниваются только отзывы одного SKU.
- ПДн маскируются, полезные отзывы не удаляются из-за ПДн.
- Порог почти точного дубля: `SequenceMatcher.ratio() >= 0.92`.
- Порядок отзывов и скрытые метки `Review` сохраняются.

---

### Task 1: Normalization, PII redaction and garbage filter

**Files:**

- Create: `src/marketplace_agent/reviews/__init__.py`
- Create: `src/marketplace_agent/reviews/cleaning.py`
- Create: `tests/reviews/test_cleaning.py`

**Interfaces:**

```python
class ReviewCleaningResult(BaseModel):
    reviews: list[Review]
    duplicate_review_ids: list[str]
    discarded_review_ids: list[str]
    redacted_review_ids: list[str]

def clean_reviews(reviews: list[Review]) -> ReviewCleaningResult: ...
```

- [x] **Step 1: Write failing tests** for HTML, spaces, PII, `норм` and `греется`.
- [x] **Step 2: Run**

```bash
uv run pytest tests/reviews/test_cleaning.py -v
```

Expected: import error for `marketplace_agent.reviews`.

- [x] **Step 3: Implement** text normalization with `html.unescape`, HTML-tag removal and whitespace collapse; mask phone, email and order-number patterns; discard only known useless texts that do not contain defect markers.
- [x] **Step 4: Run the test** and expect PASS.

### Task 2: Exact and near-duplicate removal

**Files:**

- Modify: `src/marketplace_agent/reviews/cleaning.py`
- Modify: `tests/reviews/test_cleaning.py`

- [x] **Step 1: Write failing tests**: exact and near duplicates of one SKU are removed; identical text for another SKU remains.
- [x] **Step 2: Run**

```bash
uv run pytest tests/reviews/test_cleaning.py -v
```

Expected: assertion failure because duplicates are retained.

- [x] **Step 3: Implement** comparison of normalized text with previously kept reviews of the same SKU; keep the first review and add later duplicates to `duplicate_review_ids`.
- [x] **Step 4: Run**

```bash
uv run pytest tests/reviews/test_cleaning.py -v
make check
```

Expected: all tests and Ruff pass.

- [x] **Step 5: Commit**

```bash
git add src/marketplace_agent/reviews tests/reviews ROADMAP.md
git commit -m "feat: clean and deduplicate reviews"
```