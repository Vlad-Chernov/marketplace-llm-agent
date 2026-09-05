# LangGraph Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Перенести цикл поддержки в LangGraph, сохранив публичный
интерфейс `SupportAgent.run(message, session_id, history)`.

**Architecture:** `SupportAgent` остаётся фасадом: выполняет security guards
и создаёт начальный state. `support.graph` содержит state, nodes и маршруты
LangGraph для решения LLM, вызова tool, финального ответа и эскалации.

**Tech Stack:** Python, LangGraph, Pydantic, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-09-05-langgraph-support-design.md`

## Global Constraints

- Не менять сигнатуру `SupportAgent.run`.
- Не менять `ToolRegistry`, `LLMClient`, `AgentAnswer`, `AgentDecision`.
- Не добавлять LangChain.
- Prompt injection и запрещённые запросы не запускают graph, LLM или tools.
- Максимум три вызова tool.
- Все тесты используют `FakeLLMClient`, без реального API.

---

### Task 1: Создать state и final-ветку support graph

**Files:**

- Create: `src/marketplace_agent/support/graph.py`
- Create: `tests/support/test_graph.py`

**Interfaces:**

`SupportGraphState` содержит:

```python
class SupportGraphState(TypedDict, total=False):
    messages: list[Message]
    session_id: str
    seen_calls: set[str]
    tool_steps: int
    decision: AgentDecision
    answer: AgentAnswer
    escalation_reason: str
```

`build_support_graph(registry, llm)` возвращает скомпилированный граф с
методом `.invoke(state)`.

- [ ] **Step 1: Создай тестовый файл и добавь общие helpers**

```python
from marketplace_agent.llm.base import (
    FakeLLMClient,
    LLMResponse,
    Message,
)
from marketplace_agent.support.agent import AgentAnswer
from marketplace_agent.support.graph import build_support_graph
from marketplace_agent.support.tools import ToolResult


class NoCallRegistry:
    def schemas(self) -> list[dict[str, object]]:
        return []

    def run(self, name: str, arguments: dict[str, object]) -> object:
        raise AssertionError(f"Tool must not be called: {name}")


class PolicyRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def schemas(self) -> list[dict[str, object]]:
        return [
            {
                "type": "function",
                "function": {"name": "search_policy"},
            }
        ]

    def run(
        self,
        name: str,
        arguments: dict[str, object],
    ) -> ToolResult:
        self.calls.append((name, arguments))
        return ToolResult(
            ok=True,
            data=[
                {
                    "chunk_id": "returns-01",
                    "text": "Возврат возможен в течение 14 дней.",
                }
            ],
            citations=["returns-01"],
        )


def response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="fake",
        prompt_tokens=0,
        completion_tokens=0,
    )


def initial_state() -> dict[str, object]:
    return {
        "messages": [Message(role="user", content="Вопрос")],
        "session_id": "session-001",
        "seen_calls": set(),
        "tool_steps": 0,
    }
```

- [ ] **Step 2: Добавь падающий тест final-ветки**

```python
def test_support_graph_returns_final_answer() -> None:
    graph = build_support_graph(
        registry=NoCallRegistry(),
        llm=FakeLLMClient(
            [
                response(
                    '{"kind":"final","status":"answered",'
                    '"text":"Ответ.","citations":["policy-001"]}'
                )
            ]
        ),
    )

    result = graph.invoke(initial_state())

    assert result["answer"] == AgentAnswer(
        status="answered",
        text="Ответ.",
        citations=["policy-001"],
    )
```

- [ ] **Step 3: Убедись, что тест падает**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/support/test_graph.py::test_support_graph_returns_final_answer -v
```

Ожидается ошибка импорта `marketplace_agent.support.graph`.

- [ ] **Step 4: Реализуй graph, `decide`, `finalize` и `escalate`**

В `src/marketplace_agent/support/graph.py`:

- импортируй `json`, `TypedDict`, `END`, `START`, `StateGraph`;
- импортируй `LLMClient`, `Message`, `chat_structured`,
  `StructuredOutputError`;
- импортируй `AgentAnswer`, `AgentDecision`, `ToolRegistry` из
  `marketplace_agent.support.agent`;
- создай `SupportGraphState`;
- создай `build_support_graph(registry, llm)`;
- node `decide` вызывает:

```python
decision = chat_structured(
    client=llm,
    messages=state["messages"],
    response_schema=AgentDecision,
    max_retries=0,
)
```

- при `StructuredOutputError` возвращай:

```python
{
    "escalation_reason": (
        "Не удалось безопасно обработать запрос."
    )
}
```

- node `finalize` возвращает `answer`:
  - `decision.status == "escalated"` → причина `"ambiguous_case"`;
  - `decision.status is None` или `not decision.text` → причина
    `"insufficient_data"`;
  - иначе — `AgentAnswer` из status, text и citations.
- node `escalate` возвращает:

```python
{
    "answer": AgentAnswer(
        status="escalated",
        text="Передам вопрос специалисту.",
        escalation_reason=state["escalation_reason"],
    )
}
```

- маршрут после `decide`:
  - при `escalation_reason` → `escalate`;
  - при `decision.kind == "final"` → `finalize`;
  - иначе → `execute_tool`.
- добавь edges `START → decide`, `finalize → END`, `escalate → END`.

- [ ] **Step 5: Запусти final-тест**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/support/test_graph.py::test_support_graph_returns_final_answer -v
```

Ожидается `1 passed`.

---

### Task 2: Добавить tool-ветку и ограничения

**Files:**

- Modify: `src/marketplace_agent/support/graph.py`
- Modify: `tests/support/test_graph.py`

- [ ] **Step 1: Добавь падающий тест успешного tool-вызова**

```python
def test_support_graph_calls_tool_then_returns_answer() -> None:
    registry = PolicyRegistry()
    graph = build_support_graph(
        registry=registry,
        llm=FakeLLMClient(
            [
                response(
                    '{"kind":"tool_call","tool_name":"search_policy",'
                    '"arguments":{"query":"Возврат"}}'
                ),
                response(
                    '{"kind":"final","status":"answered",'
                    '"text":"Возврат возможен 14 дней.",'
                    '"citations":["returns-01"]}'
                ),
            ]
        ),
    )

    result = graph.invoke(initial_state())

    assert registry.calls == [
        ("search_policy", {"query": "Возврат"})
    ]
    assert result["answer"].status == "answered"
    assert result["tool_steps"] == 1
```

- [ ] **Step 2: Убедись, что тест падает**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/support/test_graph.py::test_support_graph_calls_tool_then_returns_answer -v
```

Ожидается ошибка о неизвестном node `execute_tool` или неверный маршрут.

- [ ] **Step 3: Реализуй `execute_tool`**

В node:

1. Получи `decision = state["decision"]`.
2. Если `decision.tool_name is None`, верни:

```python
{"escalation_reason": "Инструмент не указан."}
```

3. Скопируй аргументы:

```python
arguments = dict(decision.arguments)
```

4. Для `get_order` добавь:

```python
arguments["session_id"] = state["session_id"]
```

5. Сформируй сигнатуру:

```python
call_signature = json.dumps(
    {
        "name": decision.tool_name,
        "arguments": arguments,
    },
    ensure_ascii=False,
    sort_keys=True,
)
```

6. При повторной сигнатуре верни:

```python
{"escalation_reason": "Инструмент вызван повторно."}
```

7. Вызови:

```python
tool_result = registry.run(decision.tool_name, arguments)
```

8. Если `not tool_result.ok`, верни:

```python
{"escalation_reason": "tool_error"}
```

9. При успехе добавь в messages:

```python
Message(
    role="user",
    content=(
        "Результат инструмента. Используй только эти "
        "данные для следующего JSON-ответа:\n"
        f"{json.dumps(tool_result.model_dump(), ensure_ascii=False)}"
    ),
)
```

10. Верни новые `messages`, `seen_calls` и `tool_steps + 1`.

После `execute_tool` добавь conditional edge:

- при `escalation_reason` → `escalate`;
- при `tool_steps == 3` → `step_limit`;
- иначе → `decide`.

Node `step_limit` возвращает:

```python
{"escalation_reason": "step_limit"}
```

После `step_limit` добавь edge в `escalate`.

- [ ] **Step 4: Добавь падающий тест повтора tool**

```python
def test_support_graph_escalates_repeated_tool_call() -> None:
    registry = PolicyRegistry()
    graph = build_support_graph(
        registry=registry,
        llm=FakeLLMClient(
            [
                response(
                    '{"kind":"tool_call","tool_name":"search_policy",'
                    '"arguments":{"query":"Возврат"}}'
                ),
                response(
                    '{"kind":"tool_call","tool_name":"search_policy",'
                    '"arguments":{"query":"Возврат"}}'
                ),
            ]
        ),
    )

    result = graph.invoke(initial_state())

    assert result["answer"].status == "escalated"
    assert result["answer"].escalation_reason == (
        "Инструмент вызван повторно."
    )
    assert registry.calls == [
        ("search_policy", {"query": "Возврат"})
    ]
```

- [ ] **Step 5: Добавь падающий тест лимита tool**

```python
def test_support_graph_escalates_after_three_tool_calls() -> None:
    registry = PolicyRegistry()
    graph = build_support_graph(
        registry=registry,
        llm=FakeLLMClient(
            [
                response(
                    '{"kind":"tool_call","tool_name":"search_policy",'
                    '"arguments":{"query":"Возврат"}}'
                ),
                response(
                    '{"kind":"tool_call","tool_name":"search_policy",'
                    '"arguments":{"query":"Гарантия"}}'
                ),
                response(
                    '{"kind":"tool_call","tool_name":"search_policy",'
                    '"arguments":{"query":"Доставка"}}'
                ),
            ]
        ),
    )

    result = graph.invoke(initial_state())

    assert result["answer"].status == "escalated"
    assert result["answer"].escalation_reason == "step_limit"
    assert len(registry.calls) == 3
```

- [ ] **Step 6: Запусти тесты graph**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/support/test_graph.py -v
```

- [ ] **Step 7: Проверь стиль**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run ruff check --fix \
  src/marketplace_agent/support/graph.py \
  tests/support/test_graph.py

git diff --check
```

---

### Task 3: Перевести `SupportAgent` на graph

**Files:**

- Modify: `src/marketplace_agent/support/agent.py`
- Modify: `tests/support/test_agent.py`

- [ ] **Step 1: Добавь тест неизменного публичного интерфейса**

```python
def test_support_agent_runs_final_answer_through_graph() -> None:
    agent = SupportAgent(
        registry=NoCallRegistry(),
        llm=FakeLLMClient(
            [
                response(
                    '{"kind":"final","status":"needs_clarification",'
                    '"text":"Укажите номер заказа.","citations":[]}'
                )
            ]
        ),
    )

    answer = agent.run(
        message="Где мой заказ?",
        session_id="session-001",
        history=[],
    )

    assert answer.status == "needs_clarification"
    assert answer.text == "Укажите номер заказа."
```

- [ ] **Step 2: В `SupportAgent.__init__` создай graph**

Импорт выполни внутри `__init__`, чтобы избежать циклического импорта:

```python
from marketplace_agent.support.graph import build_support_graph

self._graph = build_support_graph(registry, llm)
```

- [ ] **Step 3: Замени старый цикл в `run`**

Сохрани guards и `_build_messages`. После guards выполни:

```python
messages = self._build_messages(message, session_id, history)
result = self._graph.invoke(
    {
        "messages": messages,
        "session_id": session_id,
        "seen_calls": set(),
        "tool_steps": 0,
    }
)
return result["answer"]
```

Удали старый цикл с `for _ in range(3)`.

- [ ] **Step 4: Проверь фасад и безопасность**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/support/test_agent.py \
  tests/support/test_adversarial_guardrails.py -v
```

- [ ] **Step 5: Проверь graph, агент и стиль**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/support/test_graph.py \
  tests/support/test_agent.py \
  tests/support/test_adversarial_guardrails.py -v

UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run ruff check --fix \
  src/marketplace_agent/support/agent.py \
  src/marketplace_agent/support/graph.py \
  tests/support/test_agent.py \
  tests/support/test_graph.py \
  tests/support/test_adversarial_guardrails.py

git diff --check
```

- [ ] **Step 6: Закоммить graph поддержки**

```bash
git add src/marketplace_agent/support/agent.py \
  src/marketplace_agent/support/graph.py \
  tests/support/test_agent.py \
  tests/support/test_graph.py

git commit -m "feat: run support agent with LangGraph"
```

---

### Task 4: Завершить блок 12.2

**Files:**

- Modify: `ROADMAP.md`

- [ ] **Step 1: Запусти связанные тесты**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/support \
  tests/evals/test_mvp_executor.py \
  tests/evals/test_adversarial_dataset.py -v
```

- [ ] **Step 2: Проверь стиль и изменения**

```bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run ruff check \
  src/marketplace_agent/support \
  tests/support \
  tests/evals/test_mvp_executor.py \
  tests/evals/test_adversarial_dataset.py

git diff --check
```

- [ ] **Step 3: Отметь блок**

В `ROADMAP.md` замени:

```md
- [ ] **12.2:** перенести поддержку в LangGraph, сохранив интерфейсы инструментов.
```

на:

```md
- [x] **12.2:** перенести поддержку в LangGraph, сохранив интерфейсы инструментов.
```

- [ ] **Step 4: Закоммить завершение блока**

```bash
git add ROADMAP.md
git commit -m "docs: complete LangGraph support milestone"

git status --short
git log --oneline -4
```