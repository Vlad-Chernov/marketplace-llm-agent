# LangGraph support design

## Цель

Перенести внутренний цикл `SupportAgent` в LangGraph, не меняя публичный
интерфейс:

```python
SupportAgent.run(
    message: str,
    session_id: str,
    history: list[Message],
) -> AgentAnswer
```

## Границы

- `SupportAgent` остаётся точкой входа.
- `ToolRegistry` и `LLMClient` не меняются.
- Security guards выполняются до запуска графа.
- Новый модуль `marketplace_agent.support.graph` содержит orchestration.
- Максимум — три вызова tool за один запрос.
- LangChain не добавляется.

## State

`SupportGraphState` хранит:

- `messages`: историю сообщений для LLM;
- `session_id`: идентификатор сессии;
- `seen_calls`: сигнатуры уже вызванных tools;
- `tool_steps`: количество вызовов tools;
- `decision`: последний `AgentDecision` или `None`;
- `answer`: итоговый `AgentAnswer` или `None`.

## Узлы

### `decide`

Вызывает `chat_structured` с `AgentDecision` и записывает решение в state.
При `StructuredOutputError` создаёт безопасную эскалацию.

### `execute_tool`

- Проверяет, что `tool_name` указан.
- Добавляет `session_id` в аргументы `get_order`.
- Создаёт сигнатуру вызова.
- Отклоняет повторный вызов.
- Вызывает `ToolRegistry`.
- При успехе добавляет результат tool как user-context в `messages`.
- Увеличивает `tool_steps`.

### `finalize`

Преобразует корректный final-ответ модели в `AgentAnswer`.

Если модель вернула `status="escalated"`, пустой текст или отсутствующий
status, создаёт безопасную эскалацию.

### `escalate`

Создаёт:

```python
AgentAnswer(
    status="escalated",
    text="Передам вопрос специалисту.",
    escalation_reason="tool_error",
)
```

## Маршруты

```text
START
  ↓
decide
  ├─ final с корректным text → finalize → END
  ├─ tool_call и tool_steps < 3 → execute_tool → decide
  └─ ошибка / повтор / tool error / step limit → escalate → END
```

Условная ветка после `decide` выбирает final, tool call или эскалацию.

Условная ветка после `execute_tool` возвращает в `decide`, пока не достигнут
лимит в три вызова tool.

Если `tool_steps` стал равен 3, граф сразу переходит в `escalate` с
причиной `step_limit`, не делая четвёртый вызов LLM.

## Безопасность

До `graph.invoke(...)` `SupportAgent` проверяет prompt injection и
запрещённые запросы.

При срабатывании guard:

- граф не запускается;
- LLM не вызывается;
- tools не вызываются;
- возвращается `AgentAnswer(status="escalated", ...)`.

## Совместимость

Сохраняются:

- сигнатура `SupportAgent.run(...)`;
- `AgentAnswer` и `AgentDecision`;
- формат prompt и user-context с результатом tool;
- `ToolRegistry`;
- текущие причины эскалации;
- тесты поддержки и adversarial-набор.

## Проверки

- Все текущие `tests/support/test_agent.py` остаются зелёными.
- `tests/support/test_adversarial_guardrails.py` подтверждает, что опасные
  запросы не доходят до графа, LLM и tools.
- Новый тест проверяет ответ через LangGraph при неизменном
  `SupportAgent.run(...)`.
- Отдельно проверяются повтор tool и лимит из трёх вызовов.

## Вне scope

- Новые tools.
- Изменение prompt.
- Изменение моделей LLM.
- LangChain.
- Изменение формата `AgentAnswer` или `AgentDecision`.
- Перенос графа карточки: это блок 12.3.