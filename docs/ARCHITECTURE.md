# Architecture

## Goal

The system processes marketplace catalog data, reviews, and support requests through independent Python modules.

## MVP modules

1. Synthetic data generation.
2. Product-card processing pipeline.
3. Review analysis.
4. Retrieval-augmented support agent.
5. Explicit Python orchestration.
6. FastAPI service and Streamlit demonstration UI.

## Web flow

```text
Streamlit → POST /demo/content/jobs or /demo/support/jobs → FastAPI
                                                        ↓
                                             background worker + LLM
                                                        ↓
Streamlit ← GET /demo/jobs/{job_id} ← status/result/error
```

The browser never owns the long-running LLM request. It stores only the
`job_id` and polls a lightweight status endpoint, so navigating between
Streamlit sections does not cancel generation.

## Data boundaries

- `true_attributes` are available only to data generation and evaluation.
- The working system receives only noisy supplier data.
- Deterministic rules are evaluated by Python code.
- LLMs are used only for semantic tasks.
- API responses omit secrets, prompts, `true_attributes`, review text and
  hidden evaluation labels.

## Planned package layout

Source code is located in `src/marketplace_agent`.
Tests mirror the source-code structure in `tests/`.
