# Content Grounding Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a measurable barrier that rejects product-card claims unsupported by confirmed attributes or the supplier description.

**Architecture:** `validation.grounding` first compares `used_attributes` with a `GroundingEvidence` object in Python. If that check is clean, it makes one structured LLM request to evaluate free-form text against the supplier description. Both paths return the same `GroundingResult` contract; the module never receives `Product.true_attributes`.

**Tech Stack:** Python 3.11, Pydantic v2, pytest, Ruff, existing `LLMClient` and `chat_structured` helper.

## Global Constraints

- Keep `true_attributes` out of the grounding interface and prompts.
- Use `chat_structured(..., max_retries=0)` for the single semantic check.
- Do not call the LLM when `used_attributes` is unsupported.
- Do not repair or remove claims in this block; block 4.4 owns that work.
- Run tests through `uv run pytest`; run final quality checks through `make check`.

---

## File structure

- Create `src/marketplace_agent/validation/grounding.py`: Pydantic contracts, deterministic attribute comparison, LLM prompt and orchestration.
- Create `tests/validation/test_grounding.py`: observable contract tests with a local recording LLM double.
- Modify `ROADMAP.md`: mark the five completed conditions of block 4.3 after the final check passes.

### Task 1: Deterministic `used_attributes` barrier

**Files:**

- Create: `src/marketplace_agent/validation/grounding.py`
- Create: `tests/validation/test_grounding.py`

**Interfaces:**

- Consumes: `GeneratedContent` from `marketplace_agent.domain.models` and `LLMClient` from `marketplace_agent.llm.base`.
- Produces: `GroundingEvidence`, `UnsupportedClaim`, `GroundingResult`, and `validate_grounding(content, evidence, llm)` for Task 2 and the future pipeline.

- [ ] **Step 1: Write failing tests**

  These tests catch the bugs where a different attribute value or an invented attribute reaches the LLM check. Both must return the deterministic result and make zero external calls.

  Create `tests/validation/test_grounding.py`:

  ```python
  from typing import Any

  from marketplace_agent.domain.models import GeneratedContent
  from marketplace_agent.llm.base import LLMResponse
  from marketplace_agent.validation.grounding import (
      GroundingEvidence,
      validate_grounding,
  )


  class RecordingLLMClient:
      def __init__(self, content: str = '{"unsupported_claims": []}') -> None:
          self.content = content
          self.call_count = 0

      def chat(self, **_: Any) -> LLMResponse:
          self.call_count += 1
          return LLMResponse(
              content=self.content,
              model="fake-model",
              prompt_tokens=10,
              completion_tokens=5,
          )


  def make_content(used_attributes: dict[str, Any]) -> GeneratedContent:
      return GeneratedContent(
          title="Lenovo IdeaPad 14",
          bullets=["Экран 14 дюймов"],
          description="Ноутбук для учёбы и работы.",
          keywords=["ноутбук"],
          used_attributes=used_attributes,
      )


  def make_evidence() -> GroundingEvidence:
      return GroundingEvidence(
          confirmed_attributes={"screen_size": 14, "ssd_gb": 512},
          supplier_description="Lenovo IdeaPad: экран 14 дюймов, SSD 512 ГБ.",
      )


  def test_rejects_different_used_attribute_value_without_llm_call() -> None:
      client = RecordingLLMClient()

      result = validate_grounding(
          make_content({"screen_size": 14, "ssd_gb": 1024}),
          make_evidence(),
          client,
      )

      assert client.call_count == 0
      assert result.unsupported_claims == [
          {
              "field": "used_attributes",
              "claim": "ssd_gb=1024",
              "reason": "Значение не совпадает с подтверждённым: 512.",
          }
      ]


  def test_rejects_unknown_used_attribute_without_llm_call() -> None:
      client = RecordingLLMClient()

      result = validate_grounding(
          make_content({"screen_size": 14, "battery_hours": 18}),
          make_evidence(),
          client,
      )

      assert client.call_count == 0
      assert result.unsupported_claims == [
          {
              "field": "used_attributes",
              "claim": "battery_hours=18",
              "reason": "Атрибут отсутствует в подтверждённых данных.",
          }
      ]
  ```

- [ ] **Step 2: Verify the tests fail for the intended reason**

  Run:

  ```bash
  uv run pytest tests/validation/test_grounding.py -v
  ```

  Expected: collection fails with `ModuleNotFoundError: No module named 'marketplace_agent.validation.grounding'`.

- [ ] **Step 3: Implement the deterministic path**

  Create `src/marketplace_agent/validation/grounding.py`:

  ```python
  from typing import Any, Literal

  from pydantic import BaseModel, ConfigDict, Field

  from marketplace_agent.domain.models import GeneratedContent
  from marketplace_agent.llm.base import LLMClient


  class GroundingEvidence(BaseModel):
      """Evidence permitted for validating generated product content."""

      model_config = ConfigDict(extra="forbid")

      confirmed_attributes: dict[str, Any]
      supplier_description: str = Field(min_length=1)


  class UnsupportedClaim(BaseModel):
      """One product-card claim that has no permitted evidence."""

      model_config = ConfigDict(extra="forbid")

      field: Literal[
          "title", "bullets", "description", "keywords", "used_attributes"
      ]
      claim: str = Field(min_length=1)
      reason: str = Field(min_length=1)


  class GroundingResult(BaseModel):
      """Unsupported claims found in a generated product card."""

      model_config = ConfigDict(extra="forbid")

      unsupported_claims: list[UnsupportedClaim]


  def validate_grounding(
      content: GeneratedContent,
      evidence: GroundingEvidence,
      llm: LLMClient,
  ) -> GroundingResult:
      """Return unsupported structured claims before any LLM call."""

      del llm
      unsupported_claims = _validate_used_attributes(
          content.used_attributes,
          evidence.confirmed_attributes,
      )
      return GroundingResult(unsupported_claims=unsupported_claims)


  def _validate_used_attributes(
      used_attributes: dict[str, Any],
      confirmed_attributes: dict[str, Any],
  ) -> list[UnsupportedClaim]:
      unsupported_claims: list[UnsupportedClaim] = []

      for key, value in used_attributes.items():
          if key not in confirmed_attributes:
              reason = "Атрибут отсутствует в подтверждённых данных."
          elif value != confirmed_attributes[key]:
              reason = (
                  "Значение не совпадает с подтверждённым: "
                  f"{confirmed_attributes[key]!r}."
              )
          else:
              continue

          unsupported_claims.append(
              UnsupportedClaim(
                  field="used_attributes",
                  claim=f"{key}={value!r}",
                  reason=reason,
              )
          )

      return unsupported_claims
  ```

- [ ] **Step 4: Verify deterministic tests pass**

  Run:

  ```bash
  uv run pytest tests/validation/test_grounding.py -v
  ```

  Expected: `2 passed`.

### Task 2: LLM verification of free-form claims

**Files:**

- Modify: `src/marketplace_agent/validation/grounding.py`
- Modify: `tests/validation/test_grounding.py`

**Interfaces:**

- Consumes: Task 1 contracts and `chat_structured(client, messages, response_schema, max_retries)`.
- Produces: a complete `validate_grounding` implementation that invokes the LLM once only after deterministic validation passes.

- [ ] **Step 1: Add failing LLM-path tests**

  Append to `tests/validation/test_grounding.py`:

  ```python
  import pytest

  from marketplace_agent.llm.structured import StructuredOutputError


  def test_detects_invented_text_claim_in_one_llm_call() -> None:
      client = RecordingLLMClient(
          content="""
          {
            "unsupported_claims": [
              {
                "field": "description",
                "claim": "работает до 18 часов",
                "reason": "В описании поставщика нет времени автономной работы."
              }
            ]
          }
          """
      )

      result = validate_grounding(
          GeneratedContent(
              title="Lenovo IdeaPad 14",
              bullets=["Экран 14 дюймов"],
              description="Работает до 18 часов без подзарядки.",
              keywords=["ноутбук"],
              used_attributes={"screen_size": 14, "ssd_gb": 512},
          ),
          make_evidence(),
          client,
      )

      assert client.call_count == 1
      assert result.unsupported_claims == [
          {
              "field": "description",
              "claim": "работает до 18 часов",
              "reason": "В описании поставщика нет времени автономной работы.",
          }
      ]


  def test_accepts_card_when_llm_finds_no_unsupported_claims() -> None:
      client = RecordingLLMClient()

      result = validate_grounding(
          make_content({"screen_size": 14, "ssd_gb": 512}),
          make_evidence(),
          client,
      )

      assert client.call_count == 1
      assert result.unsupported_claims == []


  def test_rejects_llm_claim_with_field_outside_schema() -> None:
      client = RecordingLLMClient(
          content="""
          {
            "unsupported_claims": [
              {
                "field": "price",
                "claim": "Цена снижена",
                "reason": "Поле не разрешено схемой."
              }
            ]
          }
          """
      )

      with pytest.raises(StructuredOutputError):
          validate_grounding(
              make_content({"screen_size": 14, "ssd_gb": 512}),
              make_evidence(),
              client,
          )
  ```

- [ ] **Step 2: Verify tests fail because the LLM path is absent**

  Run:

  ```bash
  uv run pytest tests/validation/test_grounding.py -v
  ```

  Expected: the two valid LLM-path tests fail because `call_count` is `0`; the malformed-field test fails because no exception is raised.

- [ ] **Step 3: Complete the LLM path**

  Replace the imports and `validate_grounding` function in
  `src/marketplace_agent/validation/grounding.py` and append `_build_messages`.
  The rest of Task 1 stays unchanged.

  ```python
  import json
  from typing import Any, Literal

  from pydantic import BaseModel, ConfigDict, Field

  from marketplace_agent.domain.models import GeneratedContent
  from marketplace_agent.llm.base import LLMClient, Message
  from marketplace_agent.llm.structured import chat_structured
  ```

  ```python
  def validate_grounding(
      content: GeneratedContent,
      evidence: GroundingEvidence,
      llm: LLMClient,
  ) -> GroundingResult:
      """Return product-card claims unsupported by permitted evidence."""

      unsupported_claims = _validate_used_attributes(
          content.used_attributes,
          evidence.confirmed_attributes,
      )
      if unsupported_claims:
          return GroundingResult(unsupported_claims=unsupported_claims)

      return chat_structured(
          client=llm,
          messages=_build_messages(content, evidence),
          response_schema=GroundingResult,
          max_retries=0,
      )


  def _build_messages(
      content: GeneratedContent,
      evidence: GroundingEvidence,
  ) -> list[Message]:
      evidence_json = json.dumps(
          {
              "confirmed_attributes": evidence.confirmed_attributes,
              "supplier_description": evidence.supplier_description,
          },
          ensure_ascii=False,
      )
      content_json = json.dumps(
          {
              "title": content.title,
              "bullets": content.bullets,
              "description": content.description,
              "keywords": content.keywords,
          },
          ensure_ascii=False,
      )

      return [
          Message(
              role="system",
              content=(
                  "Проверь, подтверждён ли каждый атомарный факт карточки "
                  "только переданными доказательствами. Не проверяй "
                  "used_attributes: они уже проверены кодом. Верни только JSON "
                  "по схеме: {\"unsupported_claims\":[{\"field\":\"title, "
                  "bullets, description или keywords\",\"claim\":\"...\","
                  "\"reason\":\"...\"}]}. Если неподтверждённых фактов нет, "
                  "верни {\"unsupported_claims\": []}. "
                  f"Доказательства: {evidence_json}"
              ),
          ),
          Message(role="user", content=f"Карточка: {content_json}"),
      ]
  ```

- [ ] **Step 4: Verify all grounding tests pass**

  Run:

  ```bash
  uv run pytest tests/validation/test_grounding.py -v
  ```

  Expected: `5 passed`.

### Task 3: Integrate the completed learning block

**Files:**

- Modify: `ROADMAP.md`
- Verify: `src/marketplace_agent/validation/grounding.py`
- Verify: `tests/validation/test_grounding.py`

**Interfaces:**

- Consumes: working `validate_grounding(content, evidence, llm) -> GroundingResult` from Task 2.
- Produces: a tracked, validated and committed block 4.3.

- [ ] **Step 1: Mark block 4.3 complete**

  In `ROADMAP.md`, replace the five unchecked checkboxes in block 4.3 with:

  ```markdown
  - [x] Сопоставлять `used_attributes` с подтверждёнными атрибутами.
  - [x] Проверять остальные атомарные утверждения относительно описания поставщика.
  - [x] Удалять или отправлять на исправление неподтверждённые факты.
  - **Интерфейс:** `validate_grounding(content, evidence, llm) -> GroundingResult`.
  - **Результат:** отдельный измеряемый барьер против галлюцинаций.
  - **Проверка:** специально добавленный выдуманный атрибут обнаруживается.
  - **Коммит:** `feat: reject unsupported product claims`.
  ```

  Note: this block detects and returns claims for the later repair stage; it does not remove content itself.

- [ ] **Step 2: Run the full quality suite**

  Run:

  ```bash
  make check
  ```

  Expected: Ruff prints no remaining violations, and pytest reports all tests passing.

- [ ] **Step 3: Inspect the exact changes**

  Run:

  ```bash
  git diff --check
  git status --short
  ```

  Expected: only `ROADMAP.md`, `src/marketplace_agent/validation/grounding.py` and `tests/validation/test_grounding.py` are changed; `git diff --check` has no output.

- [ ] **Step 4: Commit the block**

  Run:

  ```bash
  git add ROADMAP.md src/marketplace_agent/validation/grounding.py tests/validation/test_grounding.py
  git commit -m "feat: reject unsupported product claims"
  git status --short
  ```

  Expected: Git creates the commit and the final `git status --short` has no output.

## Plan self-review

- Spec coverage: Task 1 covers confirmed-attribute comparison and early exit; Task 2 covers the one LLM call, strict schema, invented fact and safe card; Task 3 records the result and verifies the whole project.
- Consistency: all tasks use `GroundingEvidence`, `UnsupportedClaim`, `GroundingResult` and `validate_grounding(content, evidence, llm)` with the same field names.
- Scope: claim repair is explicitly deferred to block 4.4.
- No-placeholder check: no incomplete implementation steps or unspecified tests remain.
