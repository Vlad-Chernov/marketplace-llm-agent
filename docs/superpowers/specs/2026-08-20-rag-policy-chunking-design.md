# Design: RAG policy chunking

## Goal

Prepare support-policy Markdown documents for retrieval and verifiable citations.

## Scope

This block only parses documents into structured chunks. It does not implement search, embeddings, answer generation, or citation verification.

## Data model

`PolicyChunk` contains:

- `chunk_id`: stable identifier, for example `returns-01`;
- `document_id`: source filename without `.md`, for example `returns`;
- `policy_type`: policy category, equal to `document_id` in MVP;
- `heading`: Markdown section heading;
- `text`: original section text without paraphrasing.

## Parsing

`load_policy_chunks(path) -> list[PolicyChunk]` reads all Markdown files from the specified directory.

If a document contains level-two headings (`##`), each such heading starts a new chunk. The chunk includes its heading and all following text until the next level-two heading.

If a document has no level-two headings, each non-empty paragraph after the level-one document heading (`#`) becomes a chunk. Every chunk receives the document heading as its `heading`.

Chunks retain the original text so later code can verify citations against the source.

## Validation

The parser rejects:

- empty chunk text;
- duplicate `chunk_id`;
- duplicate source document IDs;
- Markdown files without non-empty policy text.

## Testing

Tests use the real support-policy corpus and verify:

- chunks are non-empty;
- chunk IDs are unique;
- metadata and headings are present;
- every chunk can be traced to its source document.

## Outcome

The project has a reproducible, structured policy corpus ready for BM25 retrieval and future cited answers.