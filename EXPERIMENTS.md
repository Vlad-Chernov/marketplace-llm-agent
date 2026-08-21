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
