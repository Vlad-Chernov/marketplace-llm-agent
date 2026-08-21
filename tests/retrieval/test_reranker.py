from marketplace_agent.retrieval.documents import PolicyChunk
from marketplace_agent.retrieval.lexical import SearchResult
from marketplace_agent.retrieval.reranker import (
    CrossEncoderReranker,
    RerankedRetriever,
)


class FakeScorer:
    def __init__(self, scores: list[float]) -> None:
        self._scores = scores
        self.pairs: list[tuple[str, str]] = []

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.pairs = pairs
        return self._scores


def make_result(chunk_id: str, rank: int) -> SearchResult:
    return SearchResult(
        chunk=PolicyChunk(
            chunk_id=chunk_id,
            document_id=chunk_id.split("-")[0],
            policy_type="returns",
            heading="Возврат",
            text=f"Текст {chunk_id}",
        ),
        score=1.0,
        rank=rank,
    )


def test_reranks_candidates_and_reassigns_ranks() -> None:
    scorer = FakeScorer([0.2, 0.9])
    reranker = CrossEncoderReranker(scorer=scorer)

    results = [
        make_result("delivery-01", rank=1),
        make_result("returns-01", rank=2),
    ]
    reranked = reranker.rerank("Как вернуть товар?", results, k=1)

    assert [result.chunk.chunk_id for result in reranked] == ["returns-01"]
    assert [(result.score, result.rank) for result in reranked] == [(0.9, 1)]
    assert scorer.pairs == [
        ("Как вернуть товар?", "Текст delivery-01"),
        ("Как вернуть товар?", "Текст returns-01"),
    ]


def test_breaks_equal_scores_by_chunk_id() -> None:
    reranker = CrossEncoderReranker(scorer=FakeScorer([0.5, 0.5]))

    reranked = reranker.rerank(
        "Вопрос",
        [
            make_result("returns-02", rank=1),
            make_result("delivery-01", rank=2),
        ],
        k=2,
    )

    assert [result.chunk.chunk_id for result in reranked] == [
        "delivery-01",
        "returns-02",
    ]
    assert [result.rank for result in reranked] == [1, 2]


def test_returns_empty_without_scoring_invalid_request() -> None:
    scorer = FakeScorer([])
    reranker = CrossEncoderReranker(scorer=scorer)

    assert reranker.rerank("", [make_result("returns-01", 1)], k=5) == []
    assert reranker.rerank("Вопрос", [], k=5) == []
    assert reranker.rerank("Вопрос", [make_result("returns-01", 1)], k=0) == []
    assert scorer.pairs == []

class RecordingRetriever:
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
        return self._results[:k]


def test_reranks_top_ten_candidates_from_base_retriever() -> None:
    base_retriever = RecordingRetriever(
        [
            make_result("delivery-01", rank=1),
            make_result("returns-01", rank=2),
        ]
    )
    reranked_retriever = RerankedRetriever(
        base_retriever=base_retriever,
        reranker=CrossEncoderReranker(scorer=FakeScorer([0.1, 0.9])),
        candidate_k=10,
    )

    results = reranked_retriever.search(
        "Как вернуть товар?",
        k=1,
        filters={"policy_type": "returns"},
    )

    assert base_retriever.calls == [
        ("Как вернуть товар?", 10, {"policy_type": "returns"})
    ]
    assert [result.chunk.chunk_id for result in results] == ["returns-01"]
    assert results[0].rank == 1