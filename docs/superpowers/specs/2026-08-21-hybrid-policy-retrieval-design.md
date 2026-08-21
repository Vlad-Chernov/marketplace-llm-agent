# Design: Hybrid policy retrieval

## Goal

Combine lexical and vector policy retrieval through Reciprocal Rank Fusion and measure all three retrievers on one fixed dataset.

## Scope

This block adds hybrid ranking, a retrieval gold dataset, Recall@5 metrics and a reproducible comparison script. It does not add reranking, LLM answer generation or citation verification.

## Hybrid retrieval

`HybridRetriever` receives `BM25Retriever` and `VectorRetriever`.

For each query it requests `candidate_k = 10` results from both retrievers with identical filters.

For every returned chunk, it calculates:

`RRF(chunk) = sum(1 / (60 + rank))`

The hybrid score is the sum of contributions from BM25 and vector results. Results are sorted by descending RRF score and then stable `chunk_id`; returned ranks start from one.

## Gold dataset

`data/gold/policy_retrieval_cases.json` contains manually reviewed cases:

- `id`;
- `query`;
- `expected_chunk_ids`;
- optional exact metadata `filters`.

Cases refer only to existing `PolicyChunk.chunk_id` values. The dataset is versioned in Git and independent of retriever implementation.

## Metrics

Recall@5 is calculated per query:

`number of expected chunk IDs retrieved in top 5 / number of expected chunk IDs`

The report uses mean Recall@5 across all cases. BM25, vector and hybrid are measured separately on exactly the same cases.

## Results and report

`evals/evaluate_retrieval.py` loads the corpus, opens the vector index, creates BM25 and hybrid retrievers, and saves per-query results in `evals/runs/`.

The script appends a Markdown table with Recall@5 for all three retrievers to `evals/REPORT.md`.

## Testing

Tests verify:

- RRF score calculation and stable ordering;
- metadata filters reach both underlying retrievers;
- dataset references valid chunks;
- Recall@5 matches a manual calculation;
- each retriever is evaluated on the same cases.

## Outcome

The project has a measured hybrid retrieval baseline and can select further search changes by evidence rather than single examples.