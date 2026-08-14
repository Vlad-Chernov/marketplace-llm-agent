# Architecture

## Goal

The system processes marketplace catalog data, reviews, and support requests through independent Python modules.

## MVP modules

1. Synthetic data generation.
2. Product-card processing pipeline.
3. Review analysis.
4. Retrieval-augmented support agent.
5. Explicit Python orchestration.

## Data boundaries

- `true_attributes` are available only to data generation and evaluation.
- The working system receives only noisy supplier data.
- Deterministic rules are evaluated by Python code.
- LLMs are used only for semantic tasks.

## Planned package layout

Source code is located in `src/marketplace_agent`.
Tests mirror the source-code structure in `tests/`.