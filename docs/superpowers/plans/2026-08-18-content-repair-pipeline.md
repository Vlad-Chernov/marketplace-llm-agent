# Content Repair Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the fixed generate → validate → repair loop from block 4.4.

**Architecture:** `content.pipeline` loads category data, extracts evidence once and owns attempts/status. `content.repair` is the isolated LLM call that edits a generated card from explicit violations. Validation remains in existing modules.

**Tech Stack:** Python 3.11, Pydantic v2, PyYAML, pytest, Ruff, existing LLM abstractions.

**Spec:** `docs/superpowers/specs/2026-08-18-content-repair-pipeline-design.md`

## Global Constraints

- Public signature: `run_content_pipeline(product, llm, max_attempts=3)`.
- Never pass `true_attributes` to the LLM.
- Max three attempts by default; duplicate cards and exhausted attempts become `manual_review`.
- Prompts live in `content/prompts/`, not business-logic strings.

---

## File structure

- Create `src/marketplace_agent/content/repair.py`: make one structured repair request.
- Create `src/marketplace_agent/content/prompts/repair_content.md`: versioned repair instructions.
- Create `src/marketplace_agent/content/pipeline.py`: fixed orchestration and result decisions.
- Create `tests/content/test_pipeline.py`: first-pass, repair, duplicate and invalid-limit tests.
- Modify `ROADMAP.md`: mark block 4.4 after final verification.

### Task 1: Repair request

**Files:** Create `content/repair.py`, create `content/prompts/repair_content.md`, test in `tests/content/test_pipeline.py`.

**Produces:**

```python
def repair_content(
    content: GeneratedContent,
    violations: list[RuleViolation],
    evidence: GroundingEvidence,
    llm: LLMClient,
) -> GeneratedContent: ...
```

- [ ] Write a test whose fake LLM returns a repaired `GeneratedContent`; assert the response is parsed and the fake receives one call.
- [ ] Run `uv run pytest tests/content/test_pipeline.py -v`; expect import failure for `content.repair`.
- [ ] Implement `repair_content` using `chat_structured(..., response_schema=GeneratedContent, max_retries=1)`. Serialize `content.model_dump()`, `[violation.model_dump() for violation in violations]`, and allowed grounding evidence into `repair_content.md` placeholders.
- [ ] Create prompt text requiring: correct only listed violations, preserve supported facts, invent nothing, return only JSON matching `GeneratedContent`.
- [ ] Re-run the test; expect it to pass.

### Task 2: Fixed pipeline

**Files:** Create `content/pipeline.py`, modify `tests/content/test_pipeline.py`.

**Produces:**

```python
def run_content_pipeline(
    product: Product,
    llm: LLMClient,
    max_attempts: int = 3,
) -> PipelineResult: ...
```

- [ ] Add a first-pass test: extraction → generation → grounding → semantic responses are valid; assert `status == "completed"`, `attempts == 1`, and no violations.
- [ ] Add a repair test: first generated title violates `title-length`, repair returns a valid card; assert `completed`, `attempts == 2` and the fake has consumed repair response.
- [ ] Add a duplicate test: generation and repair return identical invalid cards; assert `manual_review` with `attempts == 2`.
- [ ] Add an invalid-limit test: `max_attempts=0` raises `ValueError` before LLM use.
- [ ] Run the test file; expect import failure for `content.pipeline`.
- [ ] Implement pipeline helpers to load `data/specs/{category}.yaml` and `data/processed/rules.yaml`, extract attributes once, construct `GroundingEvidence`, call generator/repair, merge validators, convert each grounding claim to `RuleViolation(rule_id="unsupported-claim", severity="high")`, and fingerprint `content.model_dump_json()`.
- [ ] Run the test file; expect all tests to pass.

### Task 3: Record and verify

- [ ] Mark the four checkboxes in roadmap block 4.4 as `[x]`.
- [ ] Run `make check`; expect Ruff clean and all tests passing.
- [ ] Run `git diff --check` and inspect `git status --short`.
- [ ] Commit:

```bash
git add ROADMAP.md src/marketplace_agent/content/repair.py \
  src/marketplace_agent/content/pipeline.py \
  src/marketplace_agent/content/prompts/repair_content.md \
  tests/content/test_pipeline.py
git commit -m "feat: add generate validate repair pipeline"
```

## Plan self-review

- The plan covers one extraction, all three validators, exact repair feedback, three-attempt limit, duplicate detection and manual review.
- Prompt text is isolated; no new LLM provider interface is introduced.
- Later batch processing and `failed` handling remain outside this block.
