from marketplace_agent.evals.retrieval_metrics import (
    RetrievalCase,
    evaluate_retriever,
)
from marketplace_agent.retrieval.documents import PolicyChunk
from marketplace_agent.retrieval.lexical import SearchResult


class StubRetriever:
    def search(
        self,
        query: str,
        k: int,
        filters: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        return [
            SearchResult(
                chunk=PolicyChunk(
                    chunk_id="returns-01",
                    document_id="returns",
                    policy_type="returns",
                    heading="Возврат",
                    text="Текст",
                ),
                score=1.0,
                rank=1,
            )
        ]


def test_calculates_mean_recall_at_five_per_query() -> None:
    cases = [
        RetrievalCase(
            id="one",
            query="Первый",
            expected_chunk_ids=["returns-01"],
        ),
        RetrievalCase(
            id="two",
            query="Второй",
            expected_chunk_ids=["returns-01", "warranty-01"],
        ),
    ]

    evaluation = evaluate_retriever(StubRetriever(), cases, k=5)

    assert evaluation.recall_at_k == 0.75
    assert [result.recall_at_k for result in evaluation.results] == [
        1.0,
        0.5,
    ]


def test_calculates_mrr_latency_and_document_relevance(monkeypatch) -> None:
    clock = iter([1.0, 1.025])
    monkeypatch.setattr(
        "marketplace_agent.evals.retrieval_metrics.perf_counter",
        clock.__next__,
    )
    case = RetrievalCase(
        id="one",
        query="Вернуть товар",
        expected_chunk_ids=["returns-01"],
    )

    evaluation = evaluate_retriever(
        StubRetriever(),
        [case],
        k=5,
        relevance_key="document_id",
    )

    assert evaluation.recall_at_k == 1.0
    assert evaluation.mrr == 1.0
    assert evaluation.mean_latency_ms == 25.0
    assert evaluation.results[0].retrieved_document_ids == ["returns"]
    assert evaluation.results[0].first_relevant_rank == 1
    assert evaluation.results[0].reciprocal_rank == 1.0
    assert evaluation.results[0].latency_ms == 25.0