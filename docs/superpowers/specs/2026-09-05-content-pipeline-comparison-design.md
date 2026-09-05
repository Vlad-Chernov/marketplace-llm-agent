# Content pipeline comparison design

## Цель

Измерить пользу LangGraph для генерации карточек, сравнив на одинаковых
товарах два независимых конвейера:

- `legacy` — прежний линейный цикл;
- `langgraph` — текущий граф `generate → validate → repair`.

Сравнение проводится через реального LLM-провайдера. Результат включает
success rate, качество атрибутов, latency, токены, стоимость и типы
ошибок.

## Границы

- Production-интерфейс `run_content_pipeline(...)` остаётся графовым.
- Линейный конвейер существует только для evaluation.
- Набор состоит из 12 фиксированных шумных товаров.
- Обе версии используют одинаковые SKU, `max_attempts=3`, модель, цены,
  температуру и правила валидации.
- Prompts, extraction, repair и правила валидации не меняются.
- Ответы LLM из кэша одной версии не используются другой версией.
- Повторный запуск создаёт новые результаты и не перезаписывает прошлые.

## Компоненты

### `src/marketplace_agent/evals/legacy_content_pipeline.py`

Хранит прежнюю линейную реализацию:

```text
extract → generate → validate
                    ├─ valid → completed
                    └─ invalid → repair → validate
```

Она возвращает тот же `PipelineResult` и использует те же extraction,
generation, validation и repair-функции, что и граф. Production-код её не
импортирует.

### `src/marketplace_agent/evals/content_pipeline.py`

Расширяется до общего исполнителя двух версий.

`run_pipeline_version(...)` принимает вызываемый конвейер. Для каждого SKU
создаётся отдельный metered-клиент. Результат содержит:

- `sku`;
- `status`: `completed`, `manual_review` или `error`;
- `attempts`;
- извлечённые и использованные атрибуты;
- violations;
- `error`, если вызов завершился исключением;
- latency;
- prompt и completion tokens;
- cost;
- модель;
- безопасный case trace.

Case trace содержит только технические события: начало SKU, завершение или
исключение, статус, попытки, rule id нарушений и usage. Prompts, ответы
модели, ключи и PII в trace не сохраняются.

### `evals/compare_content_pipeline.py`

Скрипт запускает `legacy` и `langgraph` последовательно для одного manifest
и сохраняет JSON в `evals/runs/`.

Аргументы:

```text
--legacy-version <name>
--graph-version <name>
--input-price-per-million <number>
--output-price-per-million <number>
--max-attempts 3
```

После запуска скрипт печатает путь к JSON и сводку метрик обеих версий.

### Данные и отчёты

`data/gold/content_pipeline_manifest.json` расширяется до 12 фиксированных
SKU. Seed, размер каталога и порядок SKU остаются явными в manifest.

В JSON run сохраняются:

```text
run_id
created_at
manifest_sha256
provider
model
prices
max_attempts
legacy_results
graph_results
comparison
```

`EXPERIMENTS.md` получает один раздел с методикой, конфигурацией, таблицей
метрик, ссылкой на JSON и выводом.

`docs/AGENT_FAILURES.md` получает восемь отдельных разборов неуспехов из
case traces. Если в одном запуске меньше восьми неуспехов, выполняются новые
сравнения на том же manifest до накопления восьми случаев. В каждом разборе
указываются `run_id` и SKU.

## Метрики

| Метрика | Формула |
| --- | --- |
| Success rate | `completed / all cases` |
| Attribute F1 | существующая метрика extraction |
| Hallucination rate | неподтверждённые used attributes / все used attributes |
| Violations | сумма violations |
| Ошибки по типам | число `manual_review` и исключений по имени класса |
| Средняя latency | сумма latency / все cases |
| Суммарная стоимость | сумма case cost |
| Prompt/completion tokens | сумма токенов |

`manual_review` и исключение считаются неуспехом. Исключения учитываются
отдельно от violations.

SKU считается исправленным, если у графа меньше violations, либо если статус
из `error` или `manual_review` перешёл в `completed`. Обратные изменения
считаются ухудшением.

## Изоляция и воспроизводимость

- SKU загружаются только из manifest.
- Перед каждой версией создаётся отдельный LLM-клиент.
- Для версий используется отдельный cache namespace либо кэш отключается.
- В JSON сохраняются модель, provider, цены, manifest hash и время запуска.
- API-ошибка сохраняется как case result; оставшиеся SKU продолжаются.

## Проверки

Unit-тесты без API проверяют:

- одинаковый упорядоченный список SKU для legacy и graph;
- success rate, ошибки по типам и стоимость;
- `manual_review` как неуспех;
- улучшения и ухудшения по статусам, errors и violations;
- JSON с configuration и case traces;
- manifest с 12 уникальными SKU;
- отсутствие импорта legacy из production-пакета.

Live-запуск выполняется вручную с `LLM_PROGRESS=1`. Его JSON, таблица в
`EXPERIMENTS.md` и восемь разборов в `docs/AGENT_FAILURES.md` подтверждают
результат 12.4.

## Вне scope

- Изменение LangGraph-конвейера.
- Изменение prompts, правил валидации или LLM-провайдера.
- Оптимизация качества модели.
- Изменение batch processing.
- FastAPI и Streamlit из этапа 14.
