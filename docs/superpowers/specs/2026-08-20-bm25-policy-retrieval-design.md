# Design: BM25 policy retrieval

## Goal

Add a lexical retriever for support-policy chunks. It must find exact terms, policy names and numbers before vector search is introduced.

## Scope

This block implements only in-memory BM25 retrieval. It does not add embeddings, vector storage, hybrid ranking, reranking, answer generation or LLM calls.

## Data model

`SearchResult` contains:

- `chunk`: the matched `PolicyChunk`;
- `score`: BM25 relevance score;
- `rank`: one-based position in the result list.

## Normalization

Queries and chunk text use the same deterministic normalization:

- convert to lowercase with `casefold`;
- replace `ё` with `е`;
- extract Russian, Latin and numeric tokens;
- discard punctuation and whitespace.

No stemming, lemmatization or external linguistic dependency is used in MVP.

## Retrieval

`BM25Retriever(chunks)` builds an in-memory index from `PolicyChunk` objects.

`search(query, k, filters=None) -> list[SearchResult]`:

1. returns an empty list for an empty query or non-positive `k`;
2. filters chunks by exact metadata values before scoring;
3. calculates BM25 using `k1 = 1.5` and `b = 0.75`;
4. removes zero-score candidates;
5. returns at most `k` results sorted by descending score and stable chunk ID;
6. assigns ranks starting from one.

Supported filters are `document_id` and `policy_type`. Unknown filter keys match no chunks.

## Testing

Tests verify:

- an exact policy query returns the expected chunk first;
- a metadata filter restricts results to one document;
- normalization treats `ё` and `е` identically;
- results have decreasing scores and one-based ranks;
- invalid `k`, empty queries and unmatched filters return no results.

## Outcome

The project has a deterministic lexical baseline for policy retrieval. Its interface is reused by vector and hybrid retrievers in later blocks.