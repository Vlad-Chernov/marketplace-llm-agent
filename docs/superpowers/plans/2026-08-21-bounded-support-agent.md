# Bounded Support Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a three-step read-only support-agent loop with clarification and escalation.

**Architecture:** `agent.py` parses structured LLM decisions, calls only `ToolRegistry`, injects the real session ID for order lookup, and turns unsafe or incomplete paths into `AgentAnswer(status="escalated")`.

**Tech Stack:** Python, Pydantic, pytest, existing LLM and tool abstractions.

**Spec:** `docs/superpowers/specs/2026-08-21-bounded-support-agent-design.md`

### Task 1: Agent contracts and final/clarification decisions

**Files:** Create `src/marketplace_agent/support/agent.py`, `tests/support/test_agent.py`.

- [ ] Write a failing test with `FakeLLMClient` for a structured final `needs_clarification` response when the order number is absent.
- [ ] Run `uv run pytest tests/support/test_agent.py -v`; expect import failure.
- [ ] Implement `AgentDecision`, `AgentAnswer`, and `SupportAgent.run(...)` using `chat_structured`.

```python
class AgentAnswer(BaseModel):
    status: Literal["answered", "needs_clarification", "escalated"]
    text: str
    citations: list[str] = Field(default_factory=list)
    escalation_reason: str | None = None
```

- [ ] Re-run the focused test; expect PASS.

### Task 2: Registered tool calls and session injection

**Files:** Modify `src/marketplace_agent/support/agent.py`, `tests/support/test_agent.py`.

- [ ] Write failing tests for a policy tool result becoming a cited final answer and for `get_order` receiving the real session instead of model input.
- [ ] Run focused tests; expect tool-path failures.
- [ ] Implement one tool-call iteration: overwrite `session_id` only for `get_order`, call registry, append safe result to next prompt.
- [ ] Re-run focused tests; expect PASS.

### Task 3: Bounded escalation

**Files:** Modify `src/marketplace_agent/support/agent.py`, `tests/support/test_agent.py`.

- [ ] Write failing tests for an unknown tool, a repeated identical call, tool failure, and three-call limit.
- [ ] Run focused tests; expect failures.
- [ ] Track serialized `(name, arguments)` signatures; return `escalated` for every unsafe condition.
- [ ] Re-run `uv run pytest tests/support/test_agent.py -v`; expect PASS.

### Task 4: Verify and complete roadmap

- [ ] Run `make check`; expect Ruff clean and all tests passing.
- [ ] Mark block 7.4 complete in `ROADMAP.md`.
- [ ] Commit: `git add src/marketplace_agent/support/agent.py tests/support/test_agent.py ROADMAP.md && git commit -m "feat: add bounded support agent loop"`.
