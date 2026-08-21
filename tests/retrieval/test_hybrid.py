from marketplace_agent.retrieval.documents import PolicyChunk
from marketplace_agent.retrieval.hybrid import HybridRetriever
from marketplace_agent.retrieval.lexical import SearchResult


def result(chunk_id: str, rank: int) -> SearchResult:
    return SearchResult(
        chunk=PolicyChunk(
            chunk_id=chunk_id,
            document_id="policy",
            policy_type="policy",
            heading=chunk_id,
            text=chunk_id,
        ),
        score=1.0,
        rank=rank,
    )


class StubRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.calls: list[tuple[str, int, dict[str, str] | None]] = []

    def search(
        self,
        query: str,
        k: int,
        filters: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        self.calls.append((query, k, filters))
        return self._results


def test_combines_ranks_with_reciprocal_rank_fusion() -> None:
    lexical = StubRetriever([result("a", 1), result("b", 2)])
    vector = StubRetriever([result("b", 1), result("c", 2)])
    retriever = HybridRetriever(lexical, vector)

    results = retriever.search("вопрос", k=3)

    assert [item.chunk.chunk_id for item in results] == ["b", "a", "c"]
    assert results[0].score == 1 / 62 + 1 / 61
    assert [item.rank for item in results] == [1, 2, 3]


def test_passes_identical_filters_to_both_retrievers() -> None:
    lexical = StubRetriever([result("a", 1)])
    vector = StubRetriever([result("a", 1)])
    retriever = HybridRetriever(lexical, vector, candidate_k=10)

    results = retriever.search(
        "вопрос",
        k=1,
        filters={"document_id": "returns"},
    )

    assert results[0].chunk.chunk_id == "a"
    assert lexical.calls == [("вопрос", 10, {"document_id": "returns"})]
    assert vector.calls == [("вопрос", 10, {"document_id": "returns"})]