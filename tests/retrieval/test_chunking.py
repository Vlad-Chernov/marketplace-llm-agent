from marketplace_agent.retrieval.chunking import build_chunk_variants
from marketplace_agent.retrieval.documents import PolicyChunk


def make_chunk(number: int, text: str) -> PolicyChunk:
    return PolicyChunk(
        chunk_id=f"returns-{number:02d}",
        document_id="returns",
        policy_type="returns",
        heading="Возврат",
        text=text,
    )


def test_builds_unique_small_medium_and_large_variants() -> None:
    variants = build_chunk_variants(
        [
            make_chunk(1, "Первый"),
            make_chunk(2, "Второй"),
            make_chunk(3, "Третий"),
        ]
    )

    assert [chunk.chunk_id for chunk in variants["small"]] == [
        "small-returns-01",
        "small-returns-02",
        "small-returns-03",
    ]
    assert [chunk.chunk_id for chunk in variants["medium"]] == [
        "medium-returns-01",
        "medium-returns-03",
    ]
    assert [chunk.text for chunk in variants["medium"]] == [
        "Первый\n\nВторой",
        "Третий",
    ]
    assert [chunk.chunk_id for chunk in variants["large"]] == [
        "large-returns-01",
    ]
    assert [chunk.text for chunk in variants["large"]] == [
        "Первый\n\nВторой\n\nТретий",
    ]
    assert all(
        chunk.document_id == "returns"
        for chunks in variants.values()
        for chunk in chunks
    )