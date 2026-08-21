from collections import defaultdict
from collections.abc import Sequence

from marketplace_agent.retrieval.documents import PolicyChunk


def build_chunk_variants(
    chunks: Sequence[PolicyChunk],
) -> dict[str, list[PolicyChunk]]:
    """Build deterministic policy chunks of three different sizes."""

    grouped: dict[str, list[PolicyChunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.document_id].append(chunk)

    return {
        "small": _build_small(chunks),
        "medium": _build_medium(grouped),
        "large": _build_large(grouped),
    }


def _build_small(chunks: Sequence[PolicyChunk]) -> list[PolicyChunk]:
    return [
        _make_variant_chunk("small", chunk, chunk.text)
        for chunk in chunks
    ]


def _build_medium(
    grouped: dict[str, list[PolicyChunk]],
) -> list[PolicyChunk]:
    result: list[PolicyChunk] = []

    for document_chunks in grouped.values():
        for index in range(0, len(document_chunks), 2):
            pair = document_chunks[index : index + 2]
            result.append(
                _make_variant_chunk(
                    "medium",
                    pair[0],
                    "\n\n".join(chunk.text for chunk in pair),
                )
            )

    return result


def _build_large(
    grouped: dict[str, list[PolicyChunk]],
) -> list[PolicyChunk]:
    return [
        _make_variant_chunk(
            "large",
            document_chunks[0],
            "\n\n".join(chunk.text for chunk in document_chunks),
        )
        for document_chunks in grouped.values()
    ]


def _make_variant_chunk(
    variant: str,
    source: PolicyChunk,
    text: str,
) -> PolicyChunk:
    return PolicyChunk(
        chunk_id=f"{variant}-{source.chunk_id}",
        document_id=source.document_id,
        policy_type=source.policy_type,
        heading=source.heading,
        text=text,
    )