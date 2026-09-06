# Agent Failures

This document will contain at least eight analysed failure cases of the support agent.

## Case template

- Scenario
- Expected behaviour
- Actual behaviour
- Trace summary
- Root cause
- Fix or mitigation

## Run 4c07a2b4fca34ec19adf523e47d5b91f — legacy-linear-v1 — LAP-0010

- **Scenario:** генерация карточки из данных поставщика.
- **Expected behaviour:** карточка завершается статусом `completed`.
- **Actual behaviour:** `error`, `LLMProviderError`.
- **Trace summary:** `started` → `error`; попытки пайплайна не начались.
- **Root cause:** внешний LLM-запрос завершился ошибкой; безопасный trace не
  содержит техническое тело ответа провайдера.
- **Fix or mitigation:** сохранять тип ошибки и повторять эксперимент; граф в
  этом же запуске завершил SKU успешно после трёх попыток.

## Run 4c07a2b4fca34ec19adf523e47d5b91f — legacy-linear-v1 — LAP-0011

- **Scenario:** проверка и repair карточки.
- **Expected behaviour:** repair устраняет нарушения за три попытки.
- **Actual behaviour:** `manual_review` после трёх попыток.
- **Trace summary:** `started` → `manual_review`, rule `uppercase-ratio`.
- **Root cause:** модель повторно оставила слишком большую долю заглавных
  букв в заголовке.
- **Fix or mitigation:** сделать правило заглавных букв явным в repair-подсказке
  и сохранить ручную проверку как безопасный fallback.

## Run 4c07a2b4fca34ec19adf523e47d5b91f — langgraph-v1 — LAP-0011

- **Scenario:** тот же SKU в графовом конвейере.
- **Expected behaviour:** граф исправляет title либо эскалирует без публикации.
- **Actual behaviour:** `manual_review` после трёх попыток.
- **Trace summary:** `started` → `manual_review`, rule `uppercase-ratio`.
- **Root cause:** генерация/repair не устранили нарушение, а не ошибка
  маршрутизации графа.
- **Fix or mitigation:** усилить repair-инструкцию для title; текущая ветка
  `manual_review` сработала корректно и не пропустила карточку.

## Run f7dea6c0f7124febb1dfdeea4da56ac3 — legacy-linear-v1 — LAP-0005

- **Scenario:** повторный live-прогон legacy-конвейера.
- **Expected behaviour:** валидный title после repair.
- **Actual behaviour:** `manual_review` после двух попыток.
- **Trace summary:** `started` → `manual_review`, rule `uppercase-ratio`.
- **Root cause:** модель оставила нарушение формата заголовка.
- **Fix or mitigation:** добавить явное ограничение на регистр в repair и
  оставить эскалацию для неподдающихся случаев.

## Run f7dea6c0f7124febb1dfdeea4da56ac3 — legacy-linear-v1 — LAP-0007

- **Scenario:** извлечение и генерация карточки в legacy-конвейере.
- **Expected behaviour:** `completed` или управляемый retry.
- **Actual behaviour:** `error`, `LLMProviderError`.
- **Trace summary:** `started` → `error`; нет завершённой попытки пайплайна.
- **Root cause:** недоступность или transient failure внешнего провайдера;
  безопасный trace намеренно скрывает детали ответа.
- **Fix or mitigation:** продолжать retry на уровне клиента и повторять SKU в
  следующем batch-run; не публиковать частичный результат.

## Run f7dea6c0f7124febb1dfdeea4da56ac3 — legacy-linear-v1 — LAP-0010

- **Scenario:** повторный запуск SKU, завершившегося ошибкой в первом прогоне.
- **Expected behaviour:** успешная обработка после повторного запроса.
- **Actual behaviour:** снова `error`, `LLMProviderError`.
- **Trace summary:** `started` → `error`.
- **Root cause:** повторяемая внешняя ошибка провайдера в legacy-вызове; данных
  trace недостаточно, чтобы утверждать конкретный HTTP-код.
- **Fix or mitigation:** добавить отдельную метрику кодов/числа retry без
  сохранения ответа, затем повторно запускать только неуспешные SKU.

## Run f7dea6c0f7124febb1dfdeea4da56ac3 — langgraph-v1 — LAP-0003

- **Scenario:** обработка карточки графовым конвейером.
- **Expected behaviour:** `completed` или `manual_review` после исчерпания
  attempts.
- **Actual behaviour:** `error`, `LLMProviderError`.
- **Trace summary:** `started` → `error`; попытки content-цикла не зафиксированы.
- **Root cause:** запрос к внешнему LLM-провайдеру не завершился успешно.
- **Fix or mitigation:** retry в клиенте уже используется; для batch-run
  сохранять ошибку в checkpoint и перезапускать только этот SKU.

## Run f7dea6c0f7124febb1dfdeea4da56ac3 — langgraph-v1 — LAP-0008

- **Scenario:** обработка другой карточки тем же графовым конвейером.
- **Expected behaviour:** завершённый результат без потери данных.
- **Actual behaviour:** `error`, `LLMProviderError`.
- **Trace summary:** `started` → `error`; `attempts=0`.
- **Root cause:** внешняя ошибка провайдера до завершения pipeline attempt,
  а не нарушение бизнес-правил карточки.
- **Fix or mitigation:** разделять в отчётах provider failures и quality
  failures; повторять provider failure, не отправляя его сразу на ручную
  проверку контента.
