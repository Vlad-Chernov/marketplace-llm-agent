# Read-only Support Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add safe, independently testable product, order, and policy-search tools for the support agent.

**Architecture:** A shared `ToolResult` separates expected failures from exceptions. Small tool classes adapt existing repositories and the retriever; `ToolRegistry` is the only source of tool names, argument schemas, and LLM function definitions.

**Tech Stack:** Python 3.13, Pydantic v2, SQLite, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-08-21-read-only-support-tools-design.md`

## Global Constraints

- Tools are read-only and return `ToolResult` for expected failures.
- `get_order` receives `session_id` separately and never returns another user's data.
- `search_policy` returns at most three chunks and their IDs as citations.
- Tests use temporary SQLite and a stub retriever.

---

### Task 1: Tool contracts and product lookup

**Files:**
- Create: `src/marketplace_agent/support/tools.py`
- Create: `tests/support/test_tools.py`

- [ ] Write a failing test for `get_product` success, missing SKU, and blank SKU.
- [ ] Run `uv run pytest tests/support/test_tools.py -v`; expect import failure.
- [ ] Implement `ToolResult`, `Tool` protocol, `GetProductInput`, and `GetProductTool`.

```python
class ToolResult(BaseModel):
    ok: bool
    data: dict[str, Any] | list[dict[str, Any]] | None = None
    error: str | None = None
    error_code: str | None = None
    citations: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)
```

- [ ] Re-run the focused test; expect PASS.
- [ ] Commit: `git add src/marketplace_agent/support/tools.py tests/support/test_tools.py && git commit -m "feat: add product support tool"`.

### Task 2: Session-scoped order and policy search tools

**Files:**
- Modify: `src/marketplace_agent/storage/repositories.py`
- Modify: `src/marketplace_agent/support/tools.py`
- Modify: `tests/support/test_tools.py`

- [ ] Write failing tests for an owned order, absent order, foreign order, policy citations, and blank policy query.
- [ ] Run the focused tests; expect failures for missing tools.
- [ ] Add `OrderRepository.get_by_id(order_id) -> Order | None`, `GetOrderInput`, `GetOrderTool`, `SearchPolicyInput`, and `SearchPolicyTool`.

```python
if order is None:
    return ToolResult(ok=False, error="Заказ не найден.", error_code="not_found")
if order.session_id != arguments.session_id:
    return ToolResult(ok=False, error="Доступ к заказу запрещён.", error_code="access_denied")
```

- [ ] Re-run `uv run pytest tests/support/test_tools.py -v`; expect PASS.
- [ ] Commit: `git add src/marketplace_agent/storage/repositories.py src/marketplace_agent/support/tools.py tests/support/test_tools.py && git commit -m "feat: add order and policy support tools"`.

### Task 3: Registry and generated schemas

**Files:**
- Create: `src/marketplace_agent/support/registry.py`
- Create: `tests/support/test_registry.py`

- [ ] Write failing tests that reject duplicate tool names, validate bad arguments as `invalid_arguments`, and assert generated schemas contain each tool name and Pydantic input properties.
- [ ] Run `uv run pytest tests/support/test_registry.py -v`; expect import failure.
- [ ] Implement `ToolRegistry(tools)` with `run(name, arguments)` and `schemas()`.

```python
def schemas(self) -> list[dict[str, object]]:
    return [{"type": "function", "function": {"name": tool.name, "description": tool.description, "parameters": tool.input_model.model_json_schema()}} for tool in self._tools.values()]
```

- [ ] Re-run the focused registry test; expect PASS.
- [ ] Commit: `git add src/marketplace_agent/support/registry.py tests/support/test_registry.py && git commit -m "feat: register read-only support tools"`.

### Task 4: Verify and record completion

**Files:**
- Modify: `ROADMAP.md`

- [ ] Run `make check`; expect Ruff clean and all tests passing.
- [ ] Mark the four checklist items in roadmap block 7.1 as complete.
- [ ] Commit: `git add ROADMAP.md && git commit -m "docs: mark read-only support tools complete"`.
