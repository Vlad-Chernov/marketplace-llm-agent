# Blind Content Pairwise Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible blind A/B experiment comparing legacy and LangGraph product cards with a human and LLM judge.

**Architecture:** Retain completed `GeneratedContent` in the existing in-memory case result. A new evaluator selects 30 completed pairs from 40 deterministic candidates, hashes `run_id` plus `pair_id` to assign A/B, validates the human ballot, calls a structured LLM judge and calculates wins plus agreement.

**Tech Stack:** Python 3.13, Pydantic, dataclasses, pytest, Ruff and existing LLM interfaces.

**Spec:** `docs/superpowers/specs/2026-09-06-content-pairwise-judge-design.md`

## Global Constraints

- Exactly 40 candidate products; exactly 30 pairs with completed content in both versions.
- A/B order is deterministic from `run_id` and `pair_id`, not version name.
- Human choices are only `A`, `B`, `tie`, exactly once per pair.
- Ballot has only synthetic facts and A/B cards, never version names, metrics, traces or violations.
- Final JSON excludes prompts, raw responses and hidden A/B mapping.
- Production pipeline, prompts and validation rules are unchanged.
- Live calls are manual with `LLM_PROGRESS=1`; tests use `FakeLLMClient`.

---

## File Structure

- Modify `src/marketplace_agent/evals/content_pipeline.py`: retain content in memory.
- Create `src/marketplace_agent/evals/content_pairwise.py`: pairs, ballot, judge and metrics.
- Create `src/marketplace_agent/evals/prompts/judge_content_pair.md`: fixed rubric.
- Create `evals/compare_content_pairwise.py`: `prepare`/`evaluate` CLI.
- Modify `.gitignore`: ignore `evals/manual/`.
- Create `tests/evals/test_content_pairwise.py`: no-network tests.
- Modify after live run `EXPERIMENTS.md`, `ROADMAP.md`.

### Task 1: Retain generated cards

**Files:** Modify `src/marketplace_agent/evals/content_pipeline.py`; modify `tests/evals/test_content_pipeline_comparison.py`.

**Interfaces:**

```python
@dataclass(frozen=True)
class ContentPipelineCaseResult:
    content: GeneratedContent | None = None
```

Existing `evals/compare_content_pipeline.py::serialize_result` must keep omitting `content`.

- [ ] **Step 1: Write the failing test**

```python
def test_keeps_completed_generated_content() -> None:
    generated = GeneratedContent(title="Ноутбук Lenovo", bullets=["16 ГБ RAM"], description="Описание.", keywords=["ноутбук"], used_attributes={"ram_gb": "16"})
    results = run_pipeline_version(
        products=[make_product("LAP-0001")], llm_factory=FakeLLMClient,
        pipeline=lambda product, _llm, _limit: PipelineResult(sku=product.sku, content=generated, attempts=1, status="completed"),
        max_attempts=3, input_price_per_million=0.0, output_price_per_million=0.0,
    )
    assert results[0].content == generated
```

- [ ] **Step 2: Verify failure**

Run: `UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest tests/evals/test_content_pipeline_comparison.py::test_keeps_completed_generated_content -v`

Expected: FAIL because `content` is absent.

- [ ] **Step 3: Implement minimum change**

Import `GeneratedContent`, add `content: GeneratedContent | None = None` after the existing default fields, and pass `content=pipeline_result.content` in the successful `ContentPipelineCaseResult` branch.

- [ ] **Step 4: Verify**

Run: `UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest tests/evals/test_content_pipeline_comparison.py tests/evals/test_content_pipeline_execution.py -v`

Expected: PASS and current safe serializer has no `content` key.

- [ ] **Step 5: Commit**

Run: `git add src/marketplace_agent/evals/content_pipeline.py tests/evals/test_content_pipeline_comparison.py && git commit -m "feat: retain generated cards for pairwise evaluation"`

### Task 2: Build blind pairs and validate ballot

**Files:** Create `src/marketplace_agent/evals/content_pairwise.py`; create `tests/evals/test_content_pairwise.py`.

**Interfaces:**

```python
Choice = Literal["A", "B", "tie"]

@dataclass(frozen=True)
class BlindContentPair:
    pair_id: str
    sku: str
    confirmed_attributes: Mapping[str, Any]
    card_a: GeneratedContent
    card_b: GeneratedContent
    a_version: str
    b_version: str

@dataclass(frozen=True)
class HumanPairChoice:
    pair_id: str
    choice: Choice

def build_blind_pairs(legacy_results: Sequence[ContentPipelineCaseResult], graph_results: Sequence[ContentPipelineCaseResult], run_id: str, required_pair_count: int = 30) -> list[BlindContentPair]: ...
def serialize_blind_ballot(pairs: Sequence[BlindContentPair]) -> dict[str, object]: ...
def parse_human_choices(payload: Mapping[str, object], pairs: Sequence[BlindContentPair]) -> list[HumanPairChoice]: ...
```

- [ ] **Step 1: Write failing tests**

```python
def test_builds_reproducible_blind_pairs_without_version_fields() -> None:
    pairs = build_blind_pairs(completed_results("legacy"), completed_results("graph"), run_id="fixed-run", required_pair_count=2)
    ballot = serialize_blind_ballot(pairs)
    assert [pair.pair_id for pair in pairs] == ["PAIR-001", "PAIR-002"]
    assert ballot["pairs"][0]["choice"] is None
    assert "a_version" not in ballot["pairs"][0]
    assert "latency_ms" not in ballot["pairs"][0]
    assert "trace" not in ballot["pairs"][0]

def test_rejects_invalid_or_incomplete_human_choices() -> None:
    pairs = make_pairs(count=2)
    with pytest.raises(ValueError, match="exactly one choice"):
        parse_human_choices({"choices": [{"pair_id": "PAIR-001", "choice": "A"}]}, pairs)
    with pytest.raises(ValueError, match="A, B, or tie"):
        parse_human_choices({"choices": [{"pair_id": "PAIR-001", "choice": "legacy"}, {"pair_id": "PAIR-002", "choice": "B"}]}, pairs)
```

- [ ] **Step 2: Verify failure**

Run: `UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest tests/evals/test_content_pairwise.py -v`

Expected: FAIL because the module does not exist.

- [ ] **Step 3: Implement pairs and parsing**

Require identical ordered SKUs. Skip a SKU unless both statuses equal `completed` and both contents exist. Choose A with `sha256(f"{run_id}:{pair_id}".encode()).digest()[0] % 2`. If fewer than target pairs exist, raise:

```python
raise ValueError("Fewer than 30 completed content pairs are available.")
```

Visible card JSON is only `title`, `bullets`, `description`, `keywords`. Choice IDs must be exactly the pair ID set without duplicates.

- [ ] **Step 4: Verify**

Run: `UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest tests/evals/test_content_pairwise.py -v`

Expected: PASS for deterministic order, safe serialization, insufficient pairs, duplicates and invalid values.

- [ ] **Step 5: Commit**

Run: `git add src/marketplace_agent/evals/content_pairwise.py tests/evals/test_content_pairwise.py && git commit -m "feat: create blind content comparison pairs"`

### Task 3: Add LLM judge and agreement metrics

**Files:** Modify `src/marketplace_agent/evals/content_pairwise.py`; create `src/marketplace_agent/evals/prompts/judge_content_pair.md`; modify `tests/evals/test_content_pairwise.py`.

**Interfaces:**

```python
class PairJudgeDecision(BaseModel):
    choice: Choice
    reason: str = Field(min_length=1, max_length=300)

@dataclass(frozen=True)
class JudgedPair:
    pair_id: str
    choice: Choice | None
    error_type: str | None = None

@dataclass(frozen=True)
class PairwiseAgreement:
    exact_agreement: float
    cohens_kappa: float
    comparable_pair_count: int

def judge_blind_pairs(pairs: Sequence[BlindContentPair], llm_factory: Callable[[], LLMClient], progress: Callable[[dict[str, object]], None] | None = None) -> list[JudgedPair]: ...
def evaluate_pairwise_agreement(human_choices: Sequence[HumanPairChoice], judged_pairs: Sequence[JudgedPair]) -> PairwiseAgreement: ...
```

- [ ] **Step 1: Write failing tests**

```python
def test_judges_each_pair_with_new_client_and_blind_prompt() -> None:
    clients: list[RecordingFakeLLMClient] = []
    def factory() -> RecordingFakeLLMClient:
        client = RecordingFakeLLMClient([response('{"choice":"A","reason":"Точнее."}')])
        clients.append(client)
        return client
    decisions = judge_blind_pairs(make_pairs(count=2), factory)
    assert len(clients) == 2
    assert [item.choice for item in decisions] == ["A", "A"]
    assert "legacy" not in clients[0].last_messages[0].content
    assert "langgraph" not in clients[0].last_messages[0].content

def test_calculates_exact_agreement_and_kappa() -> None:
    metrics = evaluate_pairwise_agreement(
        [HumanPairChoice("PAIR-001", "A"), HumanPairChoice("PAIR-002", "B")],
        [JudgedPair("PAIR-001", "A"), JudgedPair("PAIR-002", "tie")],
    )
    assert metrics.comparable_pair_count == 2
    assert metrics.exact_agreement == pytest.approx(0.5)
    assert metrics.cohens_kappa == pytest.approx(0.0)
```

- [ ] **Step 2: Verify failure**

Run: `UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest tests/evals/test_content_pairwise.py::test_judges_each_pair_with_new_client_and_blind_prompt tests/evals/test_content_pairwise.py::test_calculates_exact_agreement_and_kappa -v`

Expected: FAIL because judge interfaces do not exist.

- [ ] **Step 3: Implement prompt, judge and metrics**

Prompt placeholders: `{confirmed_attributes}`, `{card_a}`, `{card_b}`. Rubric: factual accuracy, clarity, buyer usefulness, otherwise `tie`. Use `chat_structured` and a new client per pair. For an exception append `JudgedPair(pair_id=pair.pair_id, choice=None, error_type=type(error).__name__)`. Emit `judge_started`, `judge_completed`, `judge_error`.

For A/B/tie calculate:

```python
observed = exact_matches / count
expected = sum(human_counts[label] / count * judge_counts[label] / count for label in ("A", "B", "tie"))
kappa = 1.0 if expected == 1.0 and observed == 1.0 else (observed - expected) / (1.0 - expected)
```

Require matching unique IDs; exclude only judge errors.

- [ ] **Step 4: Verify**

Run: `UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest tests/evals/test_content_pairwise.py -v`

Expected: PASS, including judge errors and all-tie choices.

- [ ] **Step 5: Commit**

Run: `git add src/marketplace_agent/evals/content_pairwise.py src/marketplace_agent/evals/prompts/judge_content_pair.md tests/evals/test_content_pairwise.py && git commit -m "feat: judge blind content pairs"`

### Task 4: Create the two-phase CLI

**Files:** Create `evals/compare_content_pairwise.py`; modify `.gitignore`; modify `tests/evals/test_content_pairwise.py`.

**Interfaces:**

```text
uv run python evals/compare_content_pairwise.py prepare --legacy-version legacy-linear-v1 --graph-version langgraph-v1 --catalog-seed 31 --noise-seed 41 --max-attempts 3
uv run python evals/compare_content_pairwise.py evaluate --run-id <run_id> --choices evals/manual/content-pairs-<run_id>-choices.json
```

- [ ] **Step 1: Write failing serialization tests**

```python
def test_prepare_payload_uses_forty_candidates_and_thirty_pairs() -> None:
    payload = build_prepare_payload(run_id="run-001", catalog_seed=31, noise_seed=41, pairs=make_pairs(count=30), skipped_skus=["LAP-0039"])
    assert payload["candidate_count"] == 40
    assert len(payload["pairs"]) == 30
    assert payload["skipped_skus"] == ["LAP-0039"]
    assert "a_version" not in json.dumps(payload, ensure_ascii=False)

def test_final_result_excludes_prompt_and_hidden_mapping() -> None:
    text = json.dumps(serialize_pairwise_result(make_final_result()), ensure_ascii=False)
    assert "prompt" not in text
    assert "a_version" not in text
    assert "b_version" not in text
```

- [ ] **Step 2: Verify failure**

Run: `UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest tests/evals/test_content_pairwise.py -v`

Expected: FAIL because CLI helpers do not exist.

- [ ] **Step 3: Implement prepare/evaluate**

`prepare` creates 40 products with `generate_clean_products(count=40, seed=catalog_seed)`, applies `noise_product` from one `Random(noise_seed)`, runs both versions using `run_pipeline_version`, writes `evals/manual/content-pairs-<run_id>.json` with null choices, and writes its technical run record. `evaluate` validates choices, judges pairs, resolves A/B to versions in memory, then writes only aggregate safe output. Add `evals/manual/` to `.gitignore`; `LLM_PROGRESS=1` prints both pipeline and judge events.

- [ ] **Step 4: Verify**

Run: `UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run python evals/compare_content_pairwise.py --help`

Run: `UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest tests/evals/test_content_pairwise.py tests/evals/test_content_pipeline_comparison.py tests/evals/test_legacy_content_pipeline.py -v`

Expected: help shows `prepare` and `evaluate`; all tests PASS without network.

- [ ] **Step 5: Commit**

Run: `git add .gitignore evals/compare_content_pairwise.py src/marketplace_agent/evals/content_pairwise.py tests/evals/test_content_pairwise.py && git commit -m "feat: add blind content comparison CLI"`

### Task 5: Run and document E-006

**Files:** Modify after human ballot `EXPERIMENTS.md`, `ROADMAP.md`.

- [ ] **Step 1: Run quality checks**

Run: `UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run ruff check src/marketplace_agent/evals/content_pipeline.py src/marketplace_agent/evals/content_pairwise.py evals/compare_content_pairwise.py tests/evals/test_content_pairwise.py`

Run: `UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest tests/evals/test_content_pairwise.py tests/evals/test_content_pipeline_comparison.py tests/evals/test_content_pipeline_execution.py tests/evals/test_legacy_content_pipeline.py -v && git diff --check`

Expected: all checks PASS.

- [ ] **Step 2: Prepare the 30 blind pairs**

Run: `LLM_PROGRESS=1 UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run python evals/compare_content_pairwise.py prepare --legacy-version legacy-linear-v1 --graph-version langgraph-v1 --catalog-seed 31 --noise-seed 41 --max-attempts 3`

Expected: a ballot with 30 pairs; otherwise a clear insufficient-pairs error and no ballot.

- [ ] **Step 3: Evaluate manual choices**

Copy the ballot, replace each null with `"A"`, `"B"` or `"tie"`, then run: `UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run python evals/compare_content_pairwise.py evaluate --run-id <run_id> --choices evals/manual/content-pairs-<run_id>-choices.json`

Expected: human/judge wins, exact agreement, Cohen's kappa, judge errors and safe result path.

- [ ] **Step 4: Document and commit**

Add E-006 with seeds, candidate/selected counts, versions, model, human/judge wins, agreement, kappa, errors, artifact link and limitations. Mark 13.3 complete only after all 30 valid choices. Apply the spec decision rule.

Run: `git add EXPERIMENTS.md ROADMAP.md && git commit -m "evals: compare content cards with blind judges"`
