# Content Few Shot Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Measure whether two relevant gold-card examples improve LangGraph content generation relative to zero-shot generation.

**Architecture:** Static examples are validated and deterministically selected before the initial generation node. Optional examples flow through the existing pipeline and graph without changing the zero-shot default. A dedicated evaluator invokes both variants on the fixed 12-SKU manifest and compares the existing quality and operational metrics.

**Tech Stack:** Python 3.13, Pydantic v2, LangGraph, pytest, Ruff, JSON.

**Spec:** \`docs/superpowers/specs/2026-09-06-content-few-shot-design.md\`

## Global Constraints

- Only target \`confirmed_attributes\` may provide facts about the target product.
- Examples are static JSON without supplier text, PII, secrets or \`true_attributes\`.
- Defaults remain zero-shot in all public content-pipeline interfaces.
- Selector returns at most two records, ordered by \`(-score, example_id)\`.
- New run JSON contains only technical results and selected example IDs, never prompts or generated text.
- Every live command uses \`LLM_PROGRESS=1\`.
- Do not stage the personal untracked plan or \`.docx\` file.

---

## File Structure

- Create \`data/gold/content_examples.json\` — 8 static, validated laptop-card examples.
- Create \`src/marketplace_agent/content/examples.py\` — model, loader and selector.
- Modify \`src/marketplace_agent/content/generator.py\` and its Markdown prompt — optional demonstrations.
- Modify \`src/marketplace_agent/content/graph.py\` and \`pipeline.py\` — immutable graph configuration.
- Create \`src/marketplace_agent/evals/content_few_shot.py\` and \`evals/compare_content_few_shot.py\` — comparison runner and CLI.
- Create \`tests/content/test_examples.py\` and \`tests/evals/test_content_few_shot.py\`; extend existing content tests.

---

### Task 1: Validate and select gold examples

**Files:**

- Create: \`data/gold/content_examples.json\`
- Create: \`src/marketplace_agent/content/examples.py\`
- Test: \`tests/content/test_examples.py\`

**Interfaces:**

~~~python
class ContentExample(BaseModel):
    example_id: str
    category: str
    confirmed_attributes: dict[str, str | int | float | bool]
    content: GeneratedContent

def load_content_examples(path: Path | str) -> list[ContentExample]: ...

def select_similar_examples(
    category: str,
    confirmed_attributes: Mapping[str, object],
    examples: Sequence[ContentExample],
    limit: int = 2,
) -> list[ContentExample]: ...
~~~

- [ ] **Step 1: Write failing selection tests**

~~~python
def test_selects_two_examples_with_most_matching_attributes() -> None:
    selected = select_similar_examples(
        "laptop",
        {"ram_gb": "16", "storage_gb": "512"},
        [
            example("EXAMPLE-003", {"ram_gb": "8"}),
            example("EXAMPLE-002", {"ram_gb": "16"}),
            example(
                "EXAMPLE-001",
                {"ram_gb": "16", "storage_gb": "512"},
            ),
        ],
    )

    assert [item.example_id for item in selected] == [
        "EXAMPLE-001",
        "EXAMPLE-002",
    ]


def test_resolves_equal_scores_by_example_id() -> None:
    selected = select_similar_examples(
        "laptop",
        {"ram_gb": "16"},
        [
            example("EXAMPLE-002", {"ram_gb": "16"}),
            example("EXAMPLE-001", {"ram_gb": "16"}),
        ],
    )

    assert [item.example_id for item in selected] == [
        "EXAMPLE-001",
        "EXAMPLE-002",
    ]
~~~

The local \`example\` fixture must return \`ContentExample\` with a minimal valid
\`GeneratedContent\`, and its \`used_attributes\` must equal the fixture attributes.

- [ ] **Step 2: Verify the tests fail**

~~~bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/content/test_examples.py -v
~~~

Expected: collection error because \`marketplace_agent.content.examples\` does not exist.

- [ ] **Step 3: Implement model, loader and ranking**

Use \`ConfigDict(extra="forbid")\` in \`ContentExample\`. A model validator raises
\`ValueError\` when \`content.used_attributes\` has a key or value not present in
\`confirmed_attributes\`. The loader parses a JSON array, rejects duplicate IDs
with \`ValueError("Duplicate content example ID: <id>")\`, and preserves file
order. Score category equality as 100 plus one point for each exact target
attribute key/value match. Raise \`ValueError("limit must be at least 1")\` for
invalid limits.

- [ ] **Step 4: Add source data**

Create eight laptop records, \`EXAMPLE-001\` through \`EXAMPLE-008\`. Each uses
only \`ram_gb\`, \`storage_gb\`, \`screen_diagonal_in\`, \`processor\`,
\`operating_system\` and \`color\`. Vary the RAM/storage pairs among \`8/256\`,
\`8/512\`, \`16/512\`, \`16/1024\`, \`32/512\`, \`32/1024\`, \`16/256\`,
\`8/1024\`. Every record has a Russian title, exactly three short bullets,
Russian description, two keywords and \`used_attributes\` that is a subset of
its own confirmed attributes.

- [ ] **Step 5: Add invalid-data test and verify**

~~~python
def test_loader_rejects_duplicate_example_ids(tmp_path) -> None:
    path = tmp_path / "examples.json"
    path.write_text(json.dumps([record(), record()]), encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate content example ID"):
        load_content_examples(path)
~~~

~~~bash
PYTHONPATH=src UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache \
  uv run python -m json.tool data/gold/content_examples.json > /dev/null
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/content/test_examples.py -v
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run ruff check \
  src/marketplace_agent/content/examples.py tests/content/test_examples.py
git diff --check
git add data/gold/content_examples.json \
  src/marketplace_agent/content/examples.py tests/content/test_examples.py
git commit -m "feat: add content few-shot examples"
~~~

---

### Task 2: Add optional demonstrations to the generation prompt

**Files:**

- Modify: \`src/marketplace_agent/content/generator.py\`
- Modify: \`src/marketplace_agent/content/prompts/generate_content.md\`
- Test: \`tests/content/test_generator.py\`

**Interface:**

~~~python
def generate_content(
    product: Product,
    extracted_attributes: AttributeExtractionResult,
    client: LLMClient,
    examples: Sequence[ContentExample] = (),
) -> GeneratedContent: ...
~~~

- [ ] **Step 1: Write a failing prompt test**

~~~python
def test_includes_examples_as_style_only_context() -> None:
    content = generate_content(
        product,
        extracted_attributes,
        client,
        examples=[example],
    )

    assert content.title == "Ноутбук для работы"
    prompt = client.calls[0].messages[0].content
    assert "Примеры задают только стиль" in prompt
    assert "EXAMPLE-001" in prompt
~~~

Use the valid product, extraction, fake client and response pattern from
\`test_generates_content_from_confirmed_attributes\`.

- [ ] **Step 2: Verify failure**

~~~bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/content/test_generator.py::test_includes_examples_as_style_only_context -v
~~~

Expected: \`TypeError\` because \`generate_content\` does not accept \`examples\`.

- [ ] **Step 3: Implement prompt enrichment**

Add \`examples: Sequence[ContentExample] = ()\`. Build target confirmed attributes
as today, call \`select_similar_examples(product.category, confirmed_attributes,
examples)\`, and serialize each selected example as JSON containing only
\`example_id\`, \`confirmed_attributes\` and \`content\`.

Add this section before the JSON contract in \`generate_content.md\`:

~~~markdown
Примеры хороших карточек:
{examples}

Примеры задают только стиль и структуру. Факты о текущем товаре разрешено
брать только из блока «Подтверждённые характеристики». Не переноси бренды,
модели, числа или характеристики из примеров.
~~~

Format \`examples\` as \`[]\` when none are selected.

- [ ] **Step 4: Verify and commit**

~~~bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/content/test_generator.py tests/content/test_examples.py -v
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run ruff check \
  src/marketplace_agent/content/generator.py tests/content/test_generator.py
git diff --check
git add src/marketplace_agent/content/generator.py \
  src/marketplace_agent/content/prompts/generate_content.md \
  tests/content/test_generator.py
git commit -m "feat: add few-shot content generation"
~~~

---

### Task 3: Pass immutable examples through LangGraph

**Files:**

- Modify: \`src/marketplace_agent/content/graph.py\`
- Modify: \`src/marketplace_agent/content/pipeline.py\`
- Test: \`tests/content/test_content_graph.py\`
- Test: \`tests/content/test_pipeline.py\`

**Interfaces:**

~~~python
def build_content_graph(
    llm: LLMClient,
    examples: Sequence[ContentExample] = (),
): ...

def run_content_pipeline(
    product: Product,
    llm: LLMClient,
    max_attempts: int = 3,
    examples: Sequence[ContentExample] = (),
) -> PipelineResult: ...
~~~

- [ ] **Step 1: Write failing pipeline delegation test**

~~~python
def test_pipeline_passes_examples_to_content_graph(monkeypatch) -> None:
    received_examples = []

    class Graph:
        def invoke(self, state):
            return {
                "result": PipelineResult(
                    sku=state["product"].sku,
                    attempts=1,
                    status="completed",
                )
            }

    def build_graph(_llm, examples):
        received_examples.extend(examples)
        return Graph()

    monkeypatch.setattr(pipeline, "build_content_graph", build_graph)
    monkeypatch.setattr(
        pipeline,
        "extract_attributes",
        lambda *_: extraction,
    )

    result = pipeline.run_content_pipeline(
        product,
        FakeLLMClient(),
        examples=[example],
    )

    assert result.status == "completed"
    assert received_examples == [example]
~~~

Define \`product\`, \`extraction\` and \`example\` with the existing test module
fixtures.

- [ ] **Step 2: Verify failure**

~~~bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/content/test_content_graph.py::test_pipeline_passes_examples_to_content_graph -v
~~~

Expected: \`TypeError\` because \`run_content_pipeline\` has no \`examples\` parameter.

- [ ] **Step 3: Implement wiring**

Add optional examples to both public functions. Capture them in
\`build_content_graph\`, pass them to \`_generate\`, and pass them as
\`examples=examples\` to \`generate_content\`. Call
\`build_content_graph(llm, examples=examples)\` from the pipeline. Do not put
examples in \`ContentGraphState\`: they are immutable configuration.

- [ ] **Step 4: Verify and commit**

~~~bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/content/test_content_graph.py tests/content/test_pipeline.py -v
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run ruff check \
  src/marketplace_agent/content/graph.py \
  src/marketplace_agent/content/pipeline.py \
  tests/content/test_content_graph.py tests/content/test_pipeline.py
git diff --check
git add src/marketplace_agent/content/graph.py \
  src/marketplace_agent/content/pipeline.py \
  tests/content/test_content_graph.py tests/content/test_pipeline.py
git commit -m "feat: pass few-shot examples through content graph"
~~~

---

### Task 4: Compare zero-shot and few-shot runs

**Files:**

- Create: \`src/marketplace_agent/evals/content_few_shot.py\`
- Create: \`evals/compare_content_few_shot.py\`
- Test: \`tests/evals/test_content_few_shot.py\`

**Interfaces:**

~~~python
@dataclass(frozen=True)
class FewShotExperimentResult:
    comparison: ContentPipelineComparison
    zero_shot_results: list[ContentPipelineCaseResult]
    few_shot_results: list[ContentPipelineCaseResult]

def run_few_shot_comparison(
    products: Sequence[Product],
    llm_factory: Callable[[], LLMClient],
    examples: Sequence[ContentExample],
    max_attempts: int,
    input_price_per_million: float,
    output_price_per_million: float,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> FewShotExperimentResult: ...
~~~

- [ ] **Step 1: Write failing evaluator test**

~~~python
def test_runs_versions_with_separate_clients() -> None:
    created_clients = []

    def factory():
        client = FakeLLMClient()
        created_clients.append(client)
        return client

    result = run_few_shot_comparison(
        products=[product_one, product_two],
        llm_factory=factory,
        examples=[example],
        max_attempts=3,
        input_price_per_million=0.0,
        output_price_per_million=0.0,
        pipeline=completed_pipeline,
    )

    assert len(created_clients) == 4
    assert result.comparison.baseline.success_rate == 1.0
    assert result.comparison.validated.success_rate == 1.0
~~~

\`completed_pipeline\` accepts \`(product, llm, max_attempts, examples=())\` and
returns \`PipelineResult(sku=product.sku, attempts=1, status="completed")\`.

- [ ] **Step 2: Verify failure**

~~~bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/evals/test_content_few_shot.py -v
~~~

Expected: collection error because the evaluator module does not exist.

- [ ] **Step 3: Implement evaluator and CLI**

Call existing \`run_pipeline_version\` twice on the same ordered products:
\`zero-shot-v1\` passes \`examples=()\`; \`few-shot-v1\` passes loaded examples.
Pass through price values and progress callbacks. Return both raw result lists
and \`compare_pipeline_versions\` output.

The CLI accepts \`--zero-shot-version\`, \`--few-shot-version\`, optional prices
and \`--max-attempts\`; loads the existing 12-SKU manifest and gold examples;
uses \`create_llm_client(Settings())\`; prints progress only when
\`LLM_PROGRESS=1\`; and writes
\`evals/runs/content-few-shot-<run_id>.json\`. JSON contains configuration,
manifest SHA-256, example-data SHA-256, aggregate metrics, safe case results
and selected example IDs; it omits prompts, supplier text and generated text.

- [ ] **Step 4: Verify and commit**

~~~bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run pytest \
  tests/evals/test_content_few_shot.py \
  tests/evals/test_content_pipeline_comparison.py -v
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run python \
  evals/compare_content_few_shot.py --help
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache uv run ruff check \
  src/marketplace_agent/evals/content_few_shot.py \
  evals/compare_content_few_shot.py tests/evals/test_content_few_shot.py
git diff --check
git add src/marketplace_agent/evals/content_few_shot.py \
  evals/compare_content_few_shot.py tests/evals/test_content_few_shot.py
git commit -m "feat: compare zero-shot and few-shot content"
~~~

---

### Task 5: Run experiment and record decision

**Files:**

- Modify: \`EXPERIMENTS.md\`
- Modify: \`ROADMAP.md\`
- Create locally: \`evals/runs/content-few-shot-<run_id>.json\`

- [ ] **Step 1: Run local checks**

~~~bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache make check
git diff --check
~~~

- [ ] **Step 2: Run the live experiment**

~~~bash
LLM_PROGRESS=1 \
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache \
uv run python evals/compare_content_few_shot.py \
  --zero-shot-version zero-shot-v1 \
  --few-shot-version few-shot-v1 \
  --max-attempts 3
~~~

- [ ] **Step 3: Record E-004 and close 13.1**

Add E-004 with date, run ID, provider, model, manifest hash, example-data hash,
prices, both metric rows and changed SKUs. Adopt few-shot only if it does not
lower success rate or Attribute F1 and the measured quality gain justifies extra
tokens, latency and cost; otherwise retain zero-shot. Mark only 13.1 complete.

- [ ] **Step 4: Final verification and commit**

~~~bash
UV_CACHE_DIR=/private/tmp/marketplace-agent-uv-cache make check
git diff --check
git add EXPERIMENTS.md ROADMAP.md
git commit -m "evals: compare zero-shot and few-shot generation"
git status --short
~~~

