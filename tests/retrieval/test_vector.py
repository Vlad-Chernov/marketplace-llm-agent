from pathlib import Path

from marketplace_agent.retrieval.documents import PolicyChunk
from marketplace_agent.retrieval.vector import (
    VectorRetriever,
    build_vector_index,
)


class FakeEmbedder:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        normalized = text.casefold().replace("ё", "е")
        return [
        float(normalized.count("гарант")),
        float(normalized.count("доставк")),
        float(normalized.count("возврат")),
    ]


def make_chunks() -> list[PolicyChunk]:
    return [
        PolicyChunk(
            chunk_id="returns-01",
            document_id="returns",
            policy_type="returns",
            heading="Возврат",
            text="Возврат товара доступен в течение 14 дней.",
        ),
        PolicyChunk(
            chunk_id="warranty-01",
            document_id="warranty",
            policy_type="warranty",
            heading="Гарантия",
            text="Гарантия покрывает производственные дефекты.",
        ),
        PolicyChunk(
            chunk_id="delivery-01",
            document_id="delivery",
            policy_type="delivery",
            heading="Доставка",
            text="Статус доставки отображается в заказе.",
        ),
    ]


def test_builds_reopens_and_searches_vector_index(tmp_path: Path) -> None:
    index_path = tmp_path / "vector_store"
    retriever = build_vector_index(make_chunks(), index_path, FakeEmbedder())

    results = retriever.search("гарантия", k=2)
    reopened = VectorRetriever.open(index_path, FakeEmbedder())

    assert results[0].chunk.chunk_id == "warranty-01"
    assert results[0].rank == 1
    assert results[0].score > 0
    assert reopened.search("доставка", k=1)[0].chunk.chunk_id == "delivery-01"


def test_filters_vector_results_before_ranking(tmp_path: Path) -> None:
    retriever = build_vector_index(
        make_chunks(),
        tmp_path / "vector_store",
        FakeEmbedder(),
    )

    results = retriever.search(
        "возврат",
        k=3,
        filters={"document_id": "returns"},
    )

    assert [result.chunk.chunk_id for result in results] == ["returns-01"]


def test_returns_empty_for_invalid_vector_requests(tmp_path: Path) -> None:
    retriever = build_vector_index(
        make_chunks(),
        tmp_path / "vector_store",
        FakeEmbedder(),
    )

    assert retriever.search("", k=1) == []
    assert retriever.search("гарантия", k=0) == []
    assert retriever.search(
        "гарантия",
        k=1,
        filters={"unknown": "value"},
    ) == []