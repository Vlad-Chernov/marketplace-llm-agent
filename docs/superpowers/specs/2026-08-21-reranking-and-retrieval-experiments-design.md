# Reranking and retrieval experiments: design

## Goal

Choose the policy-retrieval configuration using reproducible measurements. The work covers chunk size, retrieval strategy, and reranking; it does not generate customer-facing answers.

## Scope

- Add a cross-encoder reranker after hybrid retrieval.
- Create three deterministic chunk-size variants from the same policy documents.
- Measure Recall@5, MRR, and latency per query.
- Persist per-query and aggregate results under `evals/runs/`.
- Record the conclusion of every controlled experiment in `EXPERIMENTS.md`.

## Reranker

`Reranker.rerank(query, results, k) -> list[SearchResult]` receives only candidates already returned by a retriever. It never changes the BM25 or vector indexes.

`CrossEncoderReranker` uses `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`. It scores `(query, chunk.text)` pairs for the hybrid top-10 candidates, sorts by descending score with `chunk_id` as a stable tie-breaker, and returns fresh sequential ranks. Tests use a fake scorer, so they do not download the model.

## Chunk-size variants

All variants are built deterministically from the four source policy documents:

- `small`: the current paragraph-level chunks;
- `medium`: adjacent paragraph pairs, with the final unpaired paragraph kept as one chunk;
- `large`: one full-document chunk per policy.

Each generated chunk has a unique variant-specific `chunk_id`, while retaining its original `document_id`, `policy_type`, heading, and text. Therefore chunk-size experiments judge relevance by `document_id`: the same user question remains relevant to the same policy document regardless of how that document is segmented.

The existing fixed-chunk comparison of vector, hybrid, and hybrid plus reranker continues to judge relevance by `chunk_id`. Before that comparison, every expected source ID is deterministically prefixed with `medium-`, matching the IDs of the medium index.

## Metrics and stored results

For every query, an evaluation result stores the expected identifiers, returned identifiers, first relevant rank, Recall@5, reciprocal rank, and elapsed search time in milliseconds. Aggregates contain mean Recall@5, MRR, and mean latency.

`Recall@5` is the share of expected identifiers found in the first five results. `MRR` is zero when no expected identifier appears, otherwise `1 / first_relevant_rank`.

## Controlled experiments

The evaluator runs two groups on the same golden queries:

1. Compare `hybrid-small`, `hybrid-medium`, and `hybrid-large`. Only chunk size varies; relevance is checked by `document_id`.
2. Compare `vector-medium`, `hybrid-medium`, and `hybrid-reranked-medium`. Only retrieval/reranking strategy varies; relevance is checked by `chunk_id`.

Each run writes JSON with per-query results to `evals/runs/`, nested by `chunk_size` and `strategy`. This preserves the two `hybrid-medium` measurements, which use different relevance identifiers. `EXPERIMENTS.md` receives a compact aggregate table and a written conclusion identifying the selected configuration and trade-offs.

## Error handling and tests

Invalid `k`, empty queries, and empty candidate lists return no results without invoking the cross-encoder. Missing or malformed evaluation data fails with a clear validation error.

Tests cover deterministic chunk variants, reranking and tie-breaking, MRR and latency measurement, document-level relevance, JSON persistence, and that experiment groups vary one factor at a time.

## Out of scope

- LLM answer generation and citations;
- changing the existing BM25 or vector retrieval algorithms;
- automatic production deployment of the chosen configuration.
