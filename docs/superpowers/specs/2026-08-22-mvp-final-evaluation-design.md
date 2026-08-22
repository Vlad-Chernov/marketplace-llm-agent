# MVP Final Evaluation Design

## Goal

Прогнать все 25 задач золотого набора через реальные модули MVP и сохранить измеримые результаты качества, задержек, токенов, кэша и стоимости.

## Problem

Текущий `run_eval.py` отправляет любой вопрос напрямую в LLM. Он не вызывает реальные модули карточек, отзывов, RAG и support-агента, поэтому не измеряет MVP-систему.

## Golden input

В `GoldenCase` добавляется поле `input`.

`question` остаётся пояснением для человека. `input` хранит точные структурированные данные для запуска модуля: товар, отзыв, контент карточки, `session_id` или поисковый запрос.

## Architecture

Создаётся `MvpCaseExecutor`. Он получает LLM, SQLite-репозитории, retriever и инструменты, затем выбирает реальный маршрут по `case.type`.

| Тип задачи | Реальный модуль |
| --- | --- |
| `attribute_extraction` | `extract_attributes` |
| `content_generation` | `run_content_pipeline` |
| `validation` | детерминированные и семантические валидаторы |
| `review_analysis` | `classify_review` |
| `support`, `no_answer`, `adversarial` | `SupportAgent` и read-only tools |

Для проверки `must_call_tools` используется обёртка реестра, записывающая имена вызванных инструментов. Поведение самих инструментов не меняется.

## Results and metrics

Каждая задача сохраняет prediction, raw response, модель, latency, tokens, cost, cache hit/miss и контролируемую ошибку.

Итоговая сводка содержит:

- Attribute F1 и hallucination rate;
- точность нарушений карточек;
- recall дефектов отзывов;
- success/refusal/citation metrics поддержки;
- latency p50/p95;
- суммарные токены, стоимость и cache hit rate;
- ошибки по ID задач.

В `evals/REPORT.md` добавляется версия `MVP-final` и три улучшения, выбранные после разбора ошибок.

## Constraints

- Реальные API-ключи остаются только в `.env`.
- Каждый результат сохраняется отдельно.
- `true_attributes` не передаются рабочим модулям.
- Модель не выполняет детерминированные проверки и не получает прямой доступ к базе.