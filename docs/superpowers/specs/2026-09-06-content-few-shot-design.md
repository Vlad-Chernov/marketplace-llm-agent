# Content Few Shot Experiment Design

## Purpose

Measure whether two relevant examples of good laptop cards improve the existing
LangGraph content pipeline. The experiment compares the current zero-shot
generation prompt with the same prompt enriched by selected examples.

## Scope

The change affects only content generation and its evaluation. Attribute
extraction, deterministic validation, grounding validation, semantic validation,
repair and the public default `run_content_pipeline` behaviour remain unchanged.

## Data

Create `data/gold/content_examples.json` with 8–12 static examples. Each item
has this shape:

```json
{
  "example_id": "EXAMPLE-001",
  "category": "laptop",
  "confirmed_attributes": {"ram_gb": "16", "storage_gb": "512"},
  "content": {
    "title": "Ноутбук для работы с 16 GB RAM",
    "bullets": ["16 GB RAM", "SSD 512 GB", "Компактный корпус"],
    "description": "Ноутбук для рабочих и учебных задач.",
    "keywords": ["ноутбук", "16 GB RAM"],
    "used_attributes": {"ram_gb": "16", "storage_gb": "512"}
  }
}
```

The examples are versioned source data. They contain only attributes that are
explicitly listed in the same example and never contain `true_attributes`,
supplier descriptions, API keys or personal data.

## Selection

`select_similar_examples` receives the target product category, confirmed
attributes, the static example collection and a limit. It scores each example
as follows:

- 100 points for an identical category;
- 1 point for every target attribute whose key and value exactly match the
  example's confirmed attributes.

Examples sort by descending score and then ascending `example_id`, making ties
deterministic. The generator uses exactly two selected examples. A category
mismatch is allowed only if fewer than two same-category records exist; the
current dataset contains only laptops, so normal evaluation uses laptop
examples.

## Generation interface

The existing interface becomes:

```python
def generate_content(
    product: Product,
    extracted_attributes: AttributeExtractionResult,
    client: LLMClient,
    examples: Sequence[ContentExample] = (),
) -> GeneratedContent: ...
```

`examples=()` preserves zero-shot output and all current callers. The generator
serializes the selected examples as JSON into a new `{examples}` section of the
prompt. The prompt explicitly says that examples define style only; facts about
the target may come only from its confirmed attributes.

`run_content_pipeline` and `build_content_graph` accept the same optional
example sequence and pass it only to the initial generation node. Repair uses
the existing violation-based prompt and does not receive demonstrations.

## Experiment

Use the fixed 12-SKU content-pipeline manifest. Run two isolated versions with
new LLM clients per SKU and `max_attempts=3`:

- `zero-shot-v1`: `examples=()`;
- `few-shot-v1`: loaded gold examples and deterministic selection.

Both use the same provider, model, prices, manifest, retry policy and
validation rules. Persist only technical result fields: status, attempts,
metrics, model, token usage, latency, safe errors and selected example IDs.
Never persist prompts, generated content, supplier text or `true_attributes`
in a new trace field.

Compare success rate, Attribute F1, hallucination rate, violation count,
attempts, latency, prompt/completion tokens, cost and failures by type. Few-shot
is adopted only if it does not lower success rate or Attribute F1 and its
quality benefit justifies the added latency and token cost.

## Testing

- Loader rejects duplicate IDs and an example whose `used_attributes` contains
  a value absent from `confirmed_attributes`.
- Selection ranks matching attributes first and resolves ties by ID.
- Zero-shot prompts contain no examples section; few-shot prompts contain two
  selected examples and the style-only rule.
- Pipeline default stays zero-shot; passing examples reaches the graph's
  generation node.
- Comparison runner executes both versions on matching SKUs with separate
  clients and records selected example IDs without source text.

## Failure handling

No suitable examples is valid: generation remains zero-shot. Bad gold data
fails at load time with `ValueError`. Provider failures and `manual_review`
remain existing pipeline outcomes and are reported separately from few-shot
quality metrics.
