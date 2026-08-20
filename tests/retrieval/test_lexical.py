from pathlib import Path

from marketplace_agent.retrieval.documents import (
    PolicyChunk,
    load_policy_chunks,
)
from marketplace_agent.retrieval.lexical import BM25Retriever


def test_finds_exact_policy_terms_in_expected_chunk() -> None:
    chunks = load_policy_chunks(Path("data/support"))
    retriever = BM25Retriever(chunks)

    results = retriever.search("гарантия клавиатуры", k=2)

    assert results[0].chunk.chunk_id == "warranty-02"
    assert results[0].rank == 1
    assert results[0].score > 0


def test_filters_results_by_document_id_before_ranking() -> None:
    chunks = load_policy_chunks(Path("data/support"))
    retriever = BM25Retriever(chunks)

    results = retriever.search(
        "товар",
        k=5,
        filters={"document_id": "returns"},
    )

    assert results
    assert {result.chunk.document_id for result in results} == {"returns"}


def test_normalizes_yo_character_for_search() -> None:
    chunk = PolicyChunk(
        chunk_id="sample-01",
        document_id="sample",
        policy_type="sample",
        heading="Тест",
        text="Ёмкость аккумулятора указана в характеристиках.",
    )
    retriever = BM25Retriever([chunk])

    results = retriever.search("емкость", k=1)

    assert [result.chunk.chunk_id for result in results] == ["sample-01"]


def test_assigns_ranks_and_returns_empty_for_invalid_requests() -> None:
    chunks = load_policy_chunks(Path("data/support"))
    retriever = BM25Retriever(chunks)

    results = retriever.search("гарантия", k=5)

    assert [result.rank for result in results] == list(
        range(1, len(results) + 1)
    )
    assert results == sorted(
        results,
        key=lambda result: (-result.score, result.chunk.chunk_id),
    )
    assert retriever.search("", k=5) == []
    assert retriever.search("гарантия", k=0) == []
    assert retriever.search(
        "гарантия",
        k=5,
        filters={"unknown": "value"},
    ) == []