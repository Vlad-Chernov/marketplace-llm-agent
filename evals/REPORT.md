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

Дата: 2026-08-22  
Набор: 25 golden-кейсов  
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