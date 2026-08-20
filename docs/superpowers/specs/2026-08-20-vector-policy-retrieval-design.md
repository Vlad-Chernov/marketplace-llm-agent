# Design: Vector policy retrieval

## Goal

Add reproducible semantic retrieval over support-policy chunks.

## Scope

This block adds local embeddings, a persistent ChromaDB index and metadata filtering. It does not add hybrid retrieval, reranking, answer generation or LLM calls.

## Dependencies

The project adds `chromadb` and `sentence-transformers`.

The embedding model is `intfloat/multilingual-e5-small`. It supports multilingual semantic similarity and is suitable for the local MVP.

## Index storage

The index directory is `data/vector_store/` and is excluded from Git.

`make ingest` rebuilds the index from `data/support/`. Rebuilding deletes and recreates only the `policy_chunks` collection, so the result depends only on versioned source documents and the selected model.

## Embeddings

Documents are encoded with the prefix `passage: `.

Queries are encoded with the prefix `query: `.

The Chroma collection uses cosine distance.

## Interface

`SentenceTransformerEmbedder` loads the selected local model and encodes strings.

`build_vector_index(chunks, path, embedder) -> VectorRetriever` persists chunks in the `policy_chunks` collection.

`VectorRetriever.open(path, embedder) -> VectorRetriever` opens an existing index.

`VectorRetriever.search(query, k, filters=None) -> list[SearchResult]` returns at most `k` results with one-based ranks and relevance scores.

## Metadata and filters

Each Chroma record stores:

- `chunk_id`;
- `document_id`;
- `policy_type`;
- `heading`;
- original chunk text.

Supported filters are exact `document_id` and `policy_type`. Unknown filter keys return no results. Filters are sent to Chroma before vector search.

## Testing

Tests use a deterministic fake embedder and a temporary Chroma directory. They verify:

- index creation and reopening;
- semantic query result conversion to `SearchResult`;
- metadata filtering;
- ranks and scores;
- empty query, non-positive `k` and unknown filters.

## Outcome

The project has a reproducible vector baseline that shares `SearchResult` with BM25 and can be compared or combined in later blocks.