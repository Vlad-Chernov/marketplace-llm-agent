# Policy Answers with Citations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Return support-policy answers grounded in retrieved chunks, with valid citations or a safe `no_answer`.

**Architecture:** `policy_answers.py` owns retrieval, prompt construction, structured generation, and deterministic citation checks. The LLM receives only the question and retrieved chunks; `PolicyAnswer.status` is decided by Python, not the LLM.

**Tech Stack:** Python 3.13, Pydantic v2, pytest, Ruff, existing `LLMClient` and `chat_structured`.

**Spec:** `docs/superpowers/specs/2026-08-21-policy-answers-with-citations-design.md`

## Global Constraints

- Search at most three chunks through `retriever.search(query, k=3)`.
- Never call the LLM for blank queries or an empty retrieval result.
- The LLM sees no data except the query and returned chunk ID, heading, and text.
- An answered response must cite one or more IDs from the returned chunks.
- Provider and structured-output failures return `no_answer`.

---

### Task 1: Grounded answer contract and valid answer path

**Files:**
- Create: `tests/support/test_policy_answers.py`
- Create: `src/marketplace_agent/support/__init__.py`
- Create: `src/marketplace_agent/support/policy_answers.py`
- Create: `src/marketplace_agent/support/prompts/answer_policy_question.md`

**Interfaces:**
- Consumes: `Retriever.search(query, k, filters)` and `LLMClient`.
- Produces: `PolicyAnswer(status, answer, citations)` and `answer_policy_question(query, retriever, llm) -> PolicyAnswer`.

- [ ] **Step 1: Write the failing test**

```python
from marketplace_agent.llm.base import FakeLLMClient, LLMResponse
from marketplace_agent.retrieval.documents import PolicyChunk
from marketplace_agent.retrieval.lexical import SearchResult
from marketplace_agent.support.policy_answers import answer_policy_question


class StubRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    def search(self, query: str, k: int, filters=None) -> list[SearchResult]:
        return self._results[:k]


def make_result(chunk_id: str, text: str) -> SearchResult:
    return SearchResult(PolicyChunk(chunk_id=chunk_id, document_id="returns", policy_type="returns", heading="Возврат", text=text), 1.0, 1)


def response(content: str) -> LLMResponse:
    return LLMResponse(content=content, model="fake", prompt_tokens=0, completion_tokens=0)


def test_answers_only_with_returned_policy_citation() -> None:
    answer = answer_policy_question(
        "Сколько дней можно вернуть ноутбук?",
        StubRetriever([make_result("returns-01", "Возврат возможен в течение 14 дней.")]),
        FakeLLMClient([response('{"answer":"Вернуть можно в течение 14 дней.","citations":["returns-01"]}')]),
    )

    assert answer.status == "answered"
    assert answer.citations == ["returns-01"]
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/support/test_policy_answers.py -v`  
Expected: FAIL with `ModuleNotFoundError: marketplace_agent.support`.

- [ ] **Step 3: Write minimal implementation**

```python
class PolicyAnswer(BaseModel):
    status: Literal["answered", "no_answer"]
    answer: str | None = None
    citations: list[str] = Field(default_factory=list)


def answer_policy_question(query: str, retriever: Retriever, llm: LLMClient) -> PolicyAnswer:
    results = retriever.search(query, k=3)
    if not query.strip() or not results:
        return _no_answer()
    generated = chat_structured(llm, _build_messages(query, results), GeneratedPolicyAnswer, max_retries=0)
    return _answered_or_no_answer(generated, results)
```

The prompt serializes only `chunk_id`, `heading`, and `text`, and orders the model to return Russian JSON with `answer` and `citations`.

- [ ] **Step 4: Verify GREEN**

Run: `uv run pytest tests/support/test_policy_answers.py::test_answers_only_with_returned_policy_citation -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/marketplace_agent/support tests/support/test_policy_answers.py
git commit -m "feat: answer policy questions with citations"
```

### Task 2: `no_answer` guardrails

**Files:**
- Modify: `tests/support/test_policy_answers.py`
- Modify: `src/marketplace_agent/support/policy_answers.py`

**Interfaces:**
- Consumes: a blank query, retrieval result, generated citation IDs, and `StructuredOutputError`.
- Produces: `PolicyAnswer(status="no_answer", answer=None, citations=[])` for unsupported requests.

- [ ] **Step 1: Write failing tests**

```python
def test_returns_no_answer_without_retrieval_or_llm_call() -> None:
    llm = FakeLLMClient()
    answer = answer_policy_question("Вопрос вне базы", StubRetriever([]), llm)
    assert answer.model_dump() == {"status": "no_answer", "answer": None, "citations": []}


def test_rejects_fabricated_citation() -> None:
    answer = answer_policy_question(
        "Вопрос",
        StubRetriever([make_result("returns-01", "Текст")]),
        FakeLLMClient([response('{"answer":"Ответ","citations":["warranty-99"]}')]),
    )
    assert answer.status == "no_answer"
```

- [ ] **Step 2: Verify RED**

Run: `uv run pytest tests/support/test_policy_answers.py -v`  
Expected: FAIL because fabricated citations and empty retrieval are not both guarded.

- [ ] **Step 3: Write minimal implementation**

```python
def _answered_or_no_answer(generated: GeneratedPolicyAnswer, results: Sequence[SearchResult]) -> PolicyAnswer:
    allowed_ids = {result.chunk.chunk_id for result in results}
    if not generated.answer.strip() or not generated.citations or not set(generated.citations) <= allowed_ids:
        return _no_answer()
    return PolicyAnswer(status="answered", answer=generated.answer, citations=generated.citations)
```

Wrap only `chat_structured(...)` in `try/except (StructuredOutputError, LLMProviderError)` and return `_no_answer()` when it fails.

- [ ] **Step 4: Verify GREEN**

Run: `uv run pytest tests/support/test_policy_answers.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/marketplace_agent/support/policy_answers.py tests/support/test_policy_answers.py
git commit -m "feat: reject unsupported policy answers"
```

### Task 3: Verify and complete the roadmap block

**Files:**
- Modify: `ROADMAP.md`

- [ ] **Step 1: Run project verification**

Run: `make check`  
Expected: Ruff is clean and all tests pass.

- [ ] **Step 2: Mark block 6.6 complete**

Change all three checklist items in block 6.6 from `- [ ]` to `- [x]` only after the focused and full test suites pass.

- [ ] **Step 3: Commit**

```bash
git add ROADMAP.md
git commit -m "docs: mark cited policy answers complete"
```
