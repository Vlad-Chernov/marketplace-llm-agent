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
    assert [result.recall_at_k for result in evaluation.results] == [1.0, 0.5]