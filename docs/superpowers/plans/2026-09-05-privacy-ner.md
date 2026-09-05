# Privacy NER Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Локально распознавать и маскировать PII, включая ФИО, во всех сохраняемых текстовых данных.

**Architecture:** `PiiRedactor` объединит существующие regex и spaCy NER. Экземпляр модели создаётся на время одной пакетной операции и освобождается через `close()`. Очистка отзывов, writer трасс и сохранение evaluation run используют единый санитайзер.

**Tech Stack:** Python 3.13, spaCy 3.8, `ru_core_news_sm` 3.8.0, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-05-privacy-ner-design.md`

## Global Constraints

- Никаких внешних API, токенов и ключей.
- Модель не хранится глобально.
- При недоступной NER исходный текст не сохраняется.
- Плейсхолдеры: `[PHONE]`, `[EMAIL]`, `[ORDER]`, `[ADDRESS]`, `[PERSON]`.
- Модель: `https://github.com/explosion/spacy-models/releases/download/ru_core_news_sm-3.8.0/ru_core_news_sm-3.8.0-py3-none-any.whl`.

---

### Task 1: Установить локальную NER-модель

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`

- [ ] **Step 1: Добавить зависимости**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv add \
  "spacy>=3.8.0,<3.9.0"

UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv add \
  "ru_core_news_sm @ https://github.com/explosion/spacy-models/releases/download/ru_core_news_sm-3.8.0/ru_core_news_sm-3.8.0-py3-none-any.whl"