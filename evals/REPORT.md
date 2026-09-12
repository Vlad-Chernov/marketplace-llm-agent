# Evaluation Report

## Purpose

This document reports quality metrics for the golden evaluation set.

## Planned metrics

- Attribute extraction F1
- Hallucinated attribute rate
- Product-card rule compliance
- Defect detection recall
- Retrieval Recall@5
- Support-agent success rate
- Escalation precision
- Personal-data leak count
- Latency and cost

## Results

Evaluation results will be added after the relevant modules are implemented.
## Run history

| Run ID | Suite | Version | Cases | Errors | Average latency, ms | Cost, USD | Metrics |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| cac561167a2a475a8abf5e4ff486214a | mvp | baseline-attribute-v1 | 9 | 6 | 837 | 0.000000 | attribute_f1=0.051, hallucination_rate=0.800 |
| 05cacfe5224248d095d2a92d4ab30ac7 | mvp | baseline-attribute-openrouter-v1 | 9 | 0 | 9159 | 0.000000 | attribute_f1=0.043, hallucination_rate=0.818 |
| b6100c2c24654c1b9683cea268ceb00e | mvp | baseline-attribute-strict-v1 | 9 | 0 | 12238 | 0.000000 | attribute_f1=0.222, hallucination_rate=0.000 |

## Content pipeline comparison: 40921faa48d64c1c8f8af4187fddc21d

| Version | Attribute F1 | Hallucination rate | Violations | Average latency, ms | Cost, USD |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline-no-repair | 0.000 | 0.000 | 0 | 77887 | 0.000000 |
| validated-pipeline-v1 | 0.000 | 0.000 | 0 | 39190 | 0.000000 |

Исправленные SKU: нет.
Ухудшившиеся SKU: нет.

## Content pipeline comparison: 0538b2941a304507854a902e973a5fd2

| Version | Attribute F1 | Hallucination rate | Violations | Average latency, ms | Cost, USD |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline-no-repair | 0.000 | 0.000 | 0 | 1596 | 0.000000 |
| validated-pipeline-qwen-v1 | 0.000 | 0.000 | 0 | 0 | 0.000000 |

Исправленные SKU: нет.
Ухудшившиеся SKU: нет.

## Content pipeline comparison: 8b25ddd1e76b41c3a1a8ef92f2f21316

| Version | Attribute F1 | Hallucination rate | Violations | Average latency, ms | Cost, USD |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline-no-repair | 0.971 | 0.000 | 10 | 20455 | 0.000000 |
| validated-pipeline-qwen-v2 | 0.971 | 0.000 | 1 | 46288 | 0.000000 |

Исправленные SKU: LAP-0001.
Ухудшившиеся SKU: нет.

## Review classification evaluation

Overall recall: 1.000

Mean absolute frequency error: 0.000

### Recall by defect

- battery_drain: 1.000
- keyboard_failure: 1.000
- overheating: 1.000
- screen_flicker: 1.000
- wifi_disconnect: 1.000

### Weak defects

- none

### Typical errors

- none

## Policy retrieval comparison (Recall@5)

| Retriever | Recall@k |
| --- | --- |
| bm25 | 1.000 |
| vector | 1.000 |
| hybrid | 1.000 |

### Missing chunks

## MVP final evaluation: 42316d87f5d74235a40e2e28cf68936f

Дата: 2026-08-22<br>
Набор: 25 golden-кейсов<br>
Модель: qwen/qwen3.6-27b

| Метрика | Результат |
| --- | ---: |
| Технические ошибки | 0 |
| Attribute F1 | 1.000 |
| Validation accuracy | 1.000 |
| Review recall | 1.000 |
| Support tool accuracy | 1.000 |
| Prompt tokens | 19 417 |
| Completion tokens | 5 099 |
| Latency p50 | 5 555 мс |
| Latency p95 | 16 634 мс |
| Cost | 0.000000 USD |

### Вывод

Все маршруты MVP прошли golden-набор без технических ошибок. Для support-агента подтверждены корректные вызовы инструментов: policy search, order lookup и безопасный отказ на prompt injection.

Метрики относятся к небольшому фиксированному набору. Review recall в этом MVP-run измеряется на одном кейсе, поэтому его нельзя считать полной оценкой классификатора.

### Следующие улучшения

1. Указать актуальные цены модели в `.env`, чтобы измерять реальную стоимость запусков.
2. Расширить golden-набор сложными отзывами, неоднозначными policy-вопросами и вариантами prompt injection.
3. Снизить p95 latency: добавить кэширование и ограничение времени ожидания внешнего LLM API.

## MVP final evaluation: 3a1a83ab1ea4463d9548c17915d413fe

Дата: 2026-09-05<br>
Набор: 25 golden-кейсов<br>
Модель: GigaChat-2-Pro:2.0.30.01

| Метрика | Результат |
| --- | ---: |
| Технические ошибки | 0 |
| Attribute F1 | 1.000 |
| Validation accuracy | 1.000 |
| Review recall | 1.000 |
| Support tool accuracy | 1.000 |
| Prompt tokens | 1 792 |
| Completion tokens | 1 006 |
| Latency p50 | 0 мс |
| Latency p95 | 31 778 мс |
| Cost | 0.000000 USD |

### Вывод

Все маршруты MVP прошли golden-набор без технических ошибок на GigaChat. Подтверждены корректные извлечение числовых атрибутов, semantic validation, безопасная обработка персональных данных и вызовы support-инструментов.

Стоимость указана как 0.000000 USD, потому что цены в `.env` не заданы. Значения токенов и latency учитывают кэш.
## Content pipeline comparison: 4c07a2b4fca34ec19adf523e47d5b91f

| Version | Success rate | Attribute F1 | Violations | Average latency, ms | Tokens | Cost, USD | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| legacy-linear-v1 | 0.833 | 0.914 | 1 | 99998 | 53869 | 0.000000 | LLMProviderError=1, manual_review=1 |
| langgraph-v1 | 0.917 | 0.972 | 1 | 127810 | 48484 | 0.000000 | manual_review=1 |

Исправленные SKU: LAP-0010.
Ухудшившиеся SKU: нет.

## Content pipeline comparison: f7dea6c0f7124febb1dfdeea4da56ac3

| Version | Success rate | Attribute F1 | Violations | Average latency, ms | Tokens | Cost, USD | Errors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| legacy-linear-v1 | 0.750 | 0.859 | 1 | 105535 | 57312 | 0.000000 | LLMProviderError=2, manual_review=1 |
| langgraph-v1 | 0.750 | 0.877 | 1 | 123419 | 58608 | 0.000000 | LLMProviderError=2, manual_review=1 |

Исправленные SKU: LAP-0005, LAP-0007, LAP-0010.
Ухудшившиеся SKU: LAP-0003, LAP-0008, LAP-0011.

## Blind content pairwise comparison: 232c4401044d47029c660ae2c7eb8c6d

| Метрика | Результат |
| --- | ---: |
| Кандидаты / общие успешные / пары | 50 / 33 / 30 |
| Ручные победы: LangGraph / legacy / tie | 16 / 9 / 5 |
| Победы LLM-судьи: LangGraph / legacy / tie | 10 / 15 / 5 |
| Exact agreement | 0.533 |
| Cohen's kappa | 0.264 |
| Ошибки судьи | 0 |

LLM-судья недостаточно согласуется с человеком (`κ < 0.40`) и не используется
как автоматический критерий выбора конвейера.

## Provider comparison: 25 MVP golden cases

| Провайдер | Ошибки | Attribute F1 | Validation | Review recall | Support | p50 / p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GigaChat-2-Pro | 1/25 | 1.000 | 0.833 | 1.000 | 1.000 | 7.6 / 91.6 с |
| groq/compound-mini | 17/25 (`http_429`) | — | — | — | — | 351.5 / 355.6 с |
| openrouter/free | 6/25 (`StructuredOutputError`) | 1.000 | 0.750 | 0.000 | 0.875 | 18.7 / 69.2 с |

GigaChat выбран основным провайдером. Groq и OpenRouter оставлены для будущих
экспериментов после решения квот и стабильности структурированного вывода.

## Финальная оценка: df2607258c6146a78766ad8ab80461e8

Дата: 2026-09-09<br>
Набор: 50 golden-кейсов<br>
Версия: `MVP-final`<br>
Модель: `GigaChat-2-Pro`

| Метрика | Результат |
| --- | ---: |
| Технические ошибки | 0 из 50 |
| Attribute F1 | 0.985 |
| Validation accuracy | 1.000 |
| Review recall | 1.000 |
| Support tool accuracy | 0.812 |
| Cache hits / misses | 65 / 0 |
| Prompt / completion tokens | 0 / 0 |
| Измеренная стоимость запуска | 0 ₽ |

Все 50 кейсов завершились без технических ошибок. Результаты получены из
LLM-кэша, поэтому этот запуск подтверждает воспроизводимость и корректность
конвейера, но не измеряет актуальную задержку и расход токенов внешнего API.

В последнем сравнительном некэшированном прогоне GigaChat на 50 кейсах было
израсходовано 34 119 входных и 9 753 выходных токена. При справочной цене
GigaChat 2 Pro 500 ₽ за 1 млн токенов такой прогон стоит около 21,94 ₽ вне
бесплатной квоты.

### Прогноз стоимости генерации карточек

В измеренном LangGraph-прогоне 12 карточек израсходовали 28 028 входных и
20 456 выходных токенов — в среднем около 4 040 токенов на карточку с учётом
повторных попыток.

Для сравнения выбрана модель
[`openai/gpt-oss-120b`](https://openrouter.ai/openai/gpt-oss-120b). Она подходит
по контексту и поддерживает tool calling и
[структурированные ответы](https://openrouter.ai/docs/guides/features/structured-outputs),
которые нужны агентам проекта. Опубликованный тариф OpenRouter — $0,03 за
1 млн входных и $0,17 за 1 млн выходных токенов.

| Объём | Оценка токенов | GigaChat 2 Pro | OpenRouter `gpt-oss-120b` |
| --- | ---: | ---: | ---: |
| 1 000 товаров | 4,04 млн | около 2 020 ₽ | около $0,36 (31 ₽) |
| 100 000 товаров | 404,03 млн | около 202 017 ₽ | около $35,99 (3 112 ₽) |

Рублёвый эквивалент OpenRouter рассчитан по курсу ЦБ РФ 86,473 ₽/$ на
[9 сентября 2026 года](https://www.cbr.ru/currency_base/daily/?UniDbQuery.Posted=True&UniDbQuery.To=09.09.2026).
Оценка использует опубликованную цену модели и не учитывает комиссию за покупку
кредитов OpenRouter.

Оценка не включает инфраструктуру, хранение, эмбеддинги и ручную проверку.
Фактическая сумма зависит от длины описаний, числа повторных попыток, кэша и
действующей бесплатной квоты.

Стоимость OpenRouter заметно ниже, но модель `gpt-oss-120b` ещё не проходила
наш golden-набор. Поэтому расчёт показывает потенциальную экономию, а не
подтверждает, что OpenRouter уже может заменить GigaChat по качеству и
стабильности.

### Итоговые ограничения

- Качество измерено на синтетическом golden-наборе, а не на реальном трафике.
- Текущий финальный запуск полностью обслужен кэшем.
- Support tool accuracy 0.812 ниже прежнего результата 1.000 и требует
  отдельного разбора ошибок перед production-применением.
- Локальная кластеризация отзывов при пороге 0.82 не принята в рабочий путь.
- Фоновые demo-задачи теряются при перезапуске FastAPI.
- Внешний GigaChat API иногда отвечает с высокой задержкой или сетевым
  таймаутом.
