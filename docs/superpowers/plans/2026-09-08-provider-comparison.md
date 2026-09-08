# GigaChat and Groq Provider Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add a reproducible CLI experiment that runs the 25-case MVP golden set through GigaChat and Groq without a shared cache and reports comparable quality, reliability, latency and token metrics.

**Architecture:** Reuse `GoldenCase`, `MvpCaseExecutor`, `MeteredLLMClient` and `summarize_mvp_run()` from `mvp_metrics.py`. A provider comparison module owns provider-specific settings, creates a fresh client and executor for each case, converts each run to safe aggregate output, and never persists prompts or raw responses. A CLI runs GigaChat and Groq sequentially and writes one comparison artifact.

**Tech Stack:** Python 3.13, Pydantic, dataclasses, pytest, Ruff, existing LLM clients and MVP evaluation modules.

**Spec:** `docs/superpowers/specs/2026-09-08-provider-comparison-design.md`

## Global Constraints

- Use exactly the 25 cases from `data/gold/mvp_cases.json` in source order.
- Compare `gigachat` with `GIGACHAT_MODEL` against `groq` with `GROQ_MODEL=groq/compound-mini`.
- Create a fresh LLM client and `MvpCaseExecutor` for every provider/case pair.
- Do not use `CachedLLMClient`; both runs must be uncached.
- Keep API keys in `.env`; never serialize them, prompts, raw responses, PII, hidden references or tool traces.
- A case error is recorded and must not stop the other cases or provider.
- Prices equal to zero are serialized as `cost_status: "unmeasured"`, never as free API usage.
- New tests use `FakeLLMClient`; live calls are manual only.

---

### Task 1: Provider comparison domain module

**Files:**
- Create: `src/marketplace_agent/evals/provider_comparison.py`
- Test: `tests/evals/test_provider_comparison.py`

**Interfaces:**
- Consumes: `GoldenCase`, `MvpCaseExecutor`, `Settings`, `EvaluationPrediction`, and existing metric functions.
- Produces: `ProviderRunSummary`, `ProviderComparisonResult`, `run_provider_comparison()`, and `serialize_provider_comparison()` for the CLI and documentation.

- [ ] **Step 1: Write failing tests for provider isolation and safe serialization**

Add tests with two fake providers and two minimal `GoldenCase` objects. Define local `golden_case(case_id)` and `FakeCaseExecutor` helpers in the test, plus a `ProviderSpec` factory that appends each created `FakeLLMClient` to a list. The fake case executor returns deterministic `EvaluationPrediction` values and raises on the first case for one provider. Assert that each provider receives a distinct client per case, the comparison records provider/model names, the second case still runs after the raised exception, and serialized output contains no `api_key`, `prompt`, `raw_response`, `reference` or tool trace.

```python
def test_runs_each_provider_case_with_a_fresh_client_and_safe_output() -> None:
    created: dict[str, list[FakeLLMClient]] = {"gigachat": [], "groq": []}

    def provider(name: str, model: str) -> ProviderSpec:
        def make_client() -> FakeLLMClient:
            client = FakeLLMClient()
            created[name].append(client)
            return client

        return ProviderSpec(
            name=name,
            model=model,
            api_key="test-secret",
            client_factory=make_client,
            executor_factory=lambda client: FakeCaseExecutor(client),
        )

    result = run_provider_comparison(
        cases=[golden_case("gold-001"), golden_case("gold-002")],
        providers=[provider("gigachat", "GigaChat-2-Max"), provider("groq", "groq/compound-mini")],
    )

    assert [run.provider for run in result.runs] == ["gigachat", "groq"]
    assert all(len(clients) == 2 for clients in created.values())
    text = json.dumps(serialize_provider_comparison(result))
    assert "test-secret" not in text
    assert "raw_response" not in text
    assert "reference" not in text
```

- [ ] **Step 2: Run the focused test to verify it fails**

Run:

```bash
.venv/bin/pytest tests/evals/test_provider_comparison.py -k fresh_client -v
```

Expected: FAIL because the comparison module and result types do not exist.

- [ ] **Step 3: Implement the minimal domain API**

Define immutable dataclasses with these fields:

```python
@dataclass(frozen=True)
class ProviderRunSummary:
    provider: str
    model: str
    run_id: str
    case_count: int
    error_count: int
    attribute_f1: float
    hallucination_rate: float
    validation_accuracy: float
    review_recall: float
    support_accuracy: float
    latency_p50_ms: int
    latency_p95_ms: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    cost_status: str

@dataclass(frozen=True)
class ProviderComparisonResult:
    created_at: str
    case_count: int
    dataset_sha256: str
    runs: tuple[ProviderRunSummary, ...]

def run_provider_comparison(
    cases: Sequence[GoldenCase],
    providers: Sequence[ProviderSpec],
) -> ProviderComparisonResult: ...

def serialize_provider_comparison(
    result: ProviderComparisonResult,
) -> dict[str, object]: ...
```

`ProviderSpec` contains only a provider name, model name, API key, a client factory, and an executor factory in memory. Build a fresh `MeteredLLMClient` and `MvpCaseExecutor` inside the per-case loop; do not wrap the client in `CachedLLMClient`. Catch exceptions around one case and continue. Build an `EvaluationRun` for each provider and pass it to `summarize_mvp_run()` from `src/marketplace_agent/evals/mvp_metrics.py`; map its `attribute_f1`, `validation_accuracy`, `review_recall`, `support_tool_accuracy`, token, cost and percentile fields into `ProviderRunSummary`, and calculate hallucination rate from `evaluate_attribute_extraction()` on successful attribute cases. If both configured prices are zero, set `cost_status` to `unmeasured`.

- [ ] **Step 4: Run focused tests and lint**

Run:

```bash
.venv/bin/pytest tests/evals/test_provider_comparison.py -v
.venv/bin/ruff check src/marketplace_agent/evals/provider_comparison.py tests/evals/test_provider_comparison.py
```

Expected: all focused tests pass and Ruff reports no errors.

- [ ] **Step 5: Commit the domain module**

```bash
git add src/marketplace_agent/evals/provider_comparison.py tests/evals/test_provider_comparison.py
git commit -m "feat: compare providers on MVP cases"
```

### Task 2: Provider comparison CLI

**Files:**
- Create: `evals/compare_provider_performance.py`
- Modify: `.gitignore` only if a new local artifact directory is needed
- Test: `tests/evals/test_provider_comparison.py`

**Interfaces:**
- Consumes: `run_provider_comparison()` and `serialize_provider_comparison()` from Task 1.
- Produces: `python evals/compare_provider_performance.py --help` and a JSON artifact under `evals/runs/provider-comparison-<run_id>.json`.

- [ ] **Step 1: Write failing CLI tests**

Test that argument parsing exposes `--cases` with a dataset path, that `Settings.from_environment()` is used only to obtain provider keys/models, and that the CLI writes the comparison artifact without serializing either key.

```python
def test_cli_writes_safe_provider_comparison_artifact(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(cli, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        cli,
        "run_provider_comparison",
        lambda cases, providers: ProviderComparisonResult(
            created_at="2026-09-08T00:00:00+00:00",
            case_count=len(cases),
            dataset_sha256="test-sha256",
            runs=(),
        ),
    )

    cli.main(["--cases", str(tmp_path / "mvp_cases.json")])

    payload = json.loads(next((tmp_path / "evals/runs").glob("*.json")).read_text())
    assert payload["case_count"] == 25
    assert "GROQ_API_KEY" not in json.dumps(payload)
```

- [ ] **Step 2: Run the CLI test to verify it fails**

Run:

```bash
.venv/bin/pytest tests/evals/test_provider_comparison.py -k artifact -v
```

Expected: FAIL because the CLI entry point does not exist.

- [ ] **Step 3: Implement CLI and progress output**

Load `data/gold/mvp_cases.json` by default, with an optional `--cases PATH` override, build two `ProviderSpec` values from the configured `Settings`, run the comparison in fixed order `gigachat`, `groq`, and write the safe artifact. Print one progress line per case and provider when `LLM_PROGRESS=1`; print the artifact path and one summary line per provider at the end. Refuse to start if either required key is missing, with an error naming only the missing environment variable.

- [ ] **Step 4: Verify help, tests and lint**

Run:

```bash
.venv/bin/python evals/compare_provider_performance.py --help
.venv/bin/pytest tests/evals/test_provider_comparison.py -v
.venv/bin/ruff check evals/compare_provider_performance.py src/marketplace_agent/evals/provider_comparison.py tests/evals/test_provider_comparison.py
git diff --check
```

Expected: help describes the comparison, all tests pass, Ruff is clean and diff check is empty.

- [ ] **Step 5: Commit the CLI**

```bash
git add evals/compare_provider_performance.py tests/evals/test_provider_comparison.py
git commit -m "feat: add provider comparison CLI"
```

### Task 3: Live run and reporting

**Files:**
- Modify: `EXPERIMENTS.md`
- Modify: `evals/REPORT.md`
- Modify: `DECISIONS.md`
- Modify: `ROADMAP.md`

**Interfaces:**
- Consumes: the safe artifact emitted by Task 2.
- Produces: E‑007 report, provider decision and completed roadmap item 13.4.

- [ ] **Step 1: Run the full offline verification suite**

Run:

```bash
.venv/bin/pytest tests/evals/test_provider_comparison.py tests/evals/test_mvp_final.py tests/evals/test_mvp_executor.py -v
.venv/bin/ruff check src/marketplace_agent/evals/provider_comparison.py evals/compare_provider_performance.py tests/evals/test_provider_comparison.py
git diff --check
```

- [ ] **Step 2: Run the live comparison manually**

Run with both keys present in `.env`:

```bash
LLM_PROGRESS=1 .venv/bin/python -u evals/compare_provider_performance.py
```

Expected: 25 progress results for GigaChat, then 25 for Groq, followed by the artifact path and aggregate metrics. Do not close the Mac during this run.

- [ ] **Step 3: Record the measured result**

Add E‑007 to `EXPERIMENTS.md` with the dataset hash, provider/model names, case count, errors, quality metrics, p50/p95 latency, tokens, `cost_status`, artifact link and limitations. Add the same aggregate table to `evals/REPORT.md` and a D‑004 decision to `DECISIONS.md`. Mark roadmap item 13.4 complete only if both providers completed the required routes; otherwise record it as inconclusive with the failing routes.

- [ ] **Step 4: Verify final documentation**

Run:

```bash
git diff --check
git status --short
```

Confirm only intended tracked files changed and `.env` is absent.

- [ ] **Step 5: Commit the result**

```bash
git add EXPERIMENTS.md evals/REPORT.md DECISIONS.md ROADMAP.md
git commit -m "evals: compare GigaChat and Groq"
```
