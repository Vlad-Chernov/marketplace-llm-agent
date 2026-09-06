# Журнал экспериментов

Здесь фиксируются контролируемые эксперименты: гипотеза, методика, результаты и вывод.

## Требования к журналу

- До завершения проекта должно быть не менее 5 контролируемых экспериментов.
- Для каждого эксперимента сохраняются входные данные, конфигурация, метрики и результаты по отдельным примерам.
- Эксперимент должен быть воспроизводимым.

## Эксперименты

## E-001. Валидность структурированного JSON у Groq

**Дата:** 2026-08-17

**Гипотеза:** Groq с моделью `groq/compound-mini` возвращает JSON, валидный по Pydantic-схеме, для простых задач извлечения.

**Методика:** Пять русскоязычных запросов на извлечение объёма RAM. Схема ответа: `{"ram_gb": int}`. Разрешён один повтор при невалидном JSON.

**Результаты:** 5 из 5 ответов валидны; доля валидных ответов — 1.00. Повторы не потребовались.

**Вывод:** Модель подходит для первого baseline структурированного вывода. Результат ограничен простой задачей и будет проверяться на более сложных сценариях.
## 2026-08-21 — Policy retrieval: chunk size and reranking

| Group | Configuration | Recall@5 | MRR | Mean latency, ms |
| --- | --- | ---: | ---: | ---: |
| chunk_size | hybrid-small | 1.000 | 0.958 | 17.062 |
| chunk_size | hybrid-medium | 1.000 | 0.958 | 9.357 |
| chunk_size | hybrid-large | 1.000 | 0.917 | 22.186 |
| strategy | vector-medium | 0.667 | 0.625 | 9.228 |
| strategy | hybrid-medium | 0.667 | 0.625 | 8.836 |
| strategy | hybrid-reranked-medium | 0.667 | 0.667 | 5527.779 |

Result JSON: `/Users/vlad/Documents/ChatGPT/пет проект ЛЛМ/evals/runs/retrieval-experiments-bd809b0cd67648eda14d9fea63ca4884.json`.

Decision: use `hybrid-medium` without reranker. It matches vector-medium on Recall@5 and MRR, has the lowest measured latency (8.836 ms), and avoids the 5527.779 ms reranker delay. `medium` is the chosen chunk size because it keeps Recall@5 and MRR of `small` while being faster.

## E-003. Legacy linear pipeline vs LangGraph content pipeline

**Дата:** 2026-09-05

**Гипотеза:** LangGraph-конвейер повысит долю успешно завершённых карточек
по сравнению с сохранённой линейной реализацией на одном наборе данных.

**Методика:** 12 фиксированных SKU из manifest, GigaChat-2-Pro,
`max_attempts=3`. Для каждой версии создавался новый клиент на SKU. Полные
технические traces сохранены в JSON; prompts и ответы модели не сохранялись.

**Конфигурация:** manifest SHA-256
`74f70b71096a1047f16b5c84be47e6b2595dae55ec47b8229ba4cbc3d88e1c74`.
Цены в настройках не были заданы, поэтому стоимость `$0.00` означает
«не измерена», а не «запросы были бесплатны».

| Метрика | legacy-linear-v1 | langgraph-v1 |
| --- | ---: | ---: |
| Success rate | 83.3% (10/12) | 91.7% (11/12) |
| Attribute F1 | 0.914 | 0.972 |
| Hallucination rate | 1.60% | 1.43% |
| Violations | 1 | 1 |
| Средняя задержка | 100.0 с | 127.8 с |
| Prompt / completion tokens | 34,659 / 19,210 | 28,028 / 20,456 |
| Cost USD | не измерена | не измерена |
| Failures | 1 provider error, 1 manual review | 1 manual review |

**Артефакт:** [content-pipeline-4c07a2b4fca34ec19adf523e47d5b91f.json](evals/runs/content-pipeline-4c07a2b4fca34ec19adf523e47d5b91f.json).

**Вывод:** в одном контрольном прогоне LangGraph восстановил `LAP-0010`,
который упал в legacy, и не добавил деградаций. Польза измерима по success
rate (+8.4 п.п.) и F1 (+0.058), но средняя задержка выросла примерно на 28 с.
Повторный запуск показал нестабильность внешнего провайдера: обе версии
получили success rate 75.0%; его JSON сохранён как дополнительный trace.
