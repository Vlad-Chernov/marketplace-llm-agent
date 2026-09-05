# LangGraph content pipeline design

## Цель

Перенести цикл создания карточки в LangGraph, сохранив интерфейс:

```python
run_content_pipeline(
    product: Product,
    llm: LLMClient,
    max_attempts: int = 3,
) -> PipelineResult
```

## Границы

- Извлечение атрибутов остаётся в `run_content_pipeline`.
- LangGraph управляет только генерацией, проверкой и исправлением карточки.
- `GeneratedContent`, `RuleViolation` и `PipelineResult` не меняются.
- Максимум попыток по умолчанию — три.
- `manual_review` возвращается в существующую batch-очередь без её изменения.
- LangChain не добавляется.

## State

`ContentGraphState` хранит:

- `product`: обрабатываемый товар;
- `extracted_attributes`: извлечённые характеристики;
- `evidence`: данные для grounding-проверки;
- `deterministic_rules`: детерминированные правила;
- `semantic_rules`: смысловые правила;
- `content`: текущая карточка или `None`;
- `violations`: нарушения текущей карточки;
- `attempts`: число созданных карточек;
- `max_attempts`: предел попыток;
- `seen_contents`: fingerprints ранее проверенных карточек;
- `result`: итоговый `PipelineResult` или `None`.

## Узлы

### `generate`

Создаёт первую карточку через `generate_content`, увеличивает `attempts` до 1
и записывает `content` в state.

### `validate`

Вызывает текущую `_validate_content`, сохраняет нарушения и fingerprint
`content.model_dump_json()`.

### `repair`

Вызывает `repair_content(content, violations, evidence, llm)`, увеличивает
`attempts` и записывает исправленную карточку.

### `completed`

Создаёт:

```python
PipelineResult(
    sku=product.sku,
    content=content,
    attempts=attempts,
    status="completed",
)
```

### `manual_review`

Создаёт:

```python
PipelineResult(
    sku=product.sku,
    content=content,
    violations=violations,
    attempts=attempts,
    status="manual_review",
)
```

## Маршруты

```text
START → generate → validate
                     ├─ violations пусты → completed → END
                     ├─ fingerprint уже встречался → manual_review → END
                     ├─ attempts достиг max_attempts → manual_review → END
                     └─ иначе → repair → validate
```

Проверка повторного fingerprint выполняется до добавления текущего
fingerprint в `seen_contents`. Это сохраняет текущую семантику:
повторная невалидная карточка отправляется на ручную проверку на второй
попытке.

## Совместимость

`run_content_pipeline`:

1. проверяет `max_attempts`;
2. загружает specs и правила;
3. вызывает `extract_attributes`;
4. создаёт `GroundingEvidence`;
5. передаёт подготовленный state в graph;
6. возвращает `result` из graph.

Формат LLM-вызовов и порядок вызовов сохраняются:

```text
extract → generate → grounding → semantic
```

При исправлении:

```text
repair → grounding → semantic
```

## Проверки

- Существующие `tests/content/test_pipeline.py` остаются зелёными.
- Новый тест подтверждает, что `run_content_pipeline` делегирует цикл graph.
- Отдельно проверяются:
  - успешная первая попытка;
  - исправление на второй попытке;
  - повтор невалидной карточки;
  - достижение лимита трёх попыток.
- Batch-тест подтверждает, что `manual_review` попадает в существующую
  очередь.

## Вне scope

- Изменение правил валидации.
- Изменение prompt.
- Новые LLM-провайдеры.
- Изменение batch processor.
- Сравнение метрик fixed pipeline и graph: это блок 12.4.