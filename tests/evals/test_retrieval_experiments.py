from marketplace_agent.evals.retrieval_experiments import (
    append_experiment_summary,
    build_experiment_groups,
    evaluate_experiment_groups,
)
from marketplace_agent.evals.retrieval_metrics import RetrievalCase
from marketplace_agent.retrieval.documents import PolicyChunk
from marketplace_agent.retrieval.lexical import SearchResult


def test_builds_controlled_experiment_groups() -> None:
    hybrid_small = object()
    hybrid_medium = object()
    hybrid_large = object()
    vector_medium = object()
    hybrid_reranked_medium = object()

    groups = build_experiment_groups(
        hybrid_small=hybrid_small,
        hybrid_medium=hybrid_medium,
        hybrid_large=hybrid_large,
        vector_medium=vector_medium,
        hybrid_reranked_medium=hybrid_reranked_medium,
    )

    chunk_size = groups["chunk_size"]
    strategy = groups["strategy"]

    assert [experiment.name for experiment in chunk_size] == [
        "hybrid-small",
        "hybrid-medium",
        "hybrid-large",
    ]
    assert [experiment.name for experiment in strategy] == [
        "vector-medium",
        "hybrid-medium",
        "hybrid-reranked-medium",
    ]
    assert all(
        experiment.relevance_key == "document_id"
        for experiment in chunk_size
    )
    assert all(
        experiment.relevance_key == "chunk_id"
        for experiment in strategy
    )
    assert chunk_size[1].retriever is hybrid_medium
    assert strategy[0].retriever is vector_medium
    assert strategy[2].retriever is hybrid_reranked_medium

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
                    chunk_id="medium-returns-01",
                    document_id="returns",
                    policy_type="returns",
                    heading="Возврат",
                    text="Возврат товара",
                ),
                score=1.0,
                rank=1,
            )
        ]


def test_evaluates_every_controlled_configuration() -> None:
    retriever = StubRetriever()
    groups = build_experiment_groups(
        hybrid_small=retriever,
        hybrid_medium=retriever,
        hybrid_large=retriever,
        vector_medium=retriever,
        hybrid_reranked_medium=retriever,
    )
    cases = [
        RetrievalCase(
            id="return",
            query="Как вернуть товар?",
            expected_chunk_ids=["returns-01"],
        )
    ]

    evaluations = evaluate_experiment_groups(groups, cases, k=5)

    assert set(evaluations) == {"chunk_size", "strategy"}
    assert set(evaluations["chunk_size"]) == {
        "hybrid-small",
        "hybrid-medium",
        "hybrid-large",
    }
    assert set(evaluations["strategy"]) == {
        "vector-medium",
        "hybrid-medium",
        "hybrid-reranked-medium",
    }
    assert (
        evaluations["chunk_size"]["hybrid-medium"].results[0].relevance_key
        == "document_id"
    )
    assert evaluations["chunk_size"]["hybrid-medium"].recall_at_k == 1.0
    assert (
        evaluations["strategy"]["hybrid-medium"].results[0].relevance_key
        == "chunk_id"
    )
    assert evaluations["strategy"]["vector-medium"].recall_at_k == 1.0

def test_appends_experiment_summary_with_real_measurements(
    tmp_path,
) -> None:
    retriever = StubRetriever()
    groups = build_experiment_groups(
        hybrid_small=retriever,
        hybrid_medium=retriever,
        hybrid_large=retriever,
        vector_medium=retriever,
        hybrid_reranked_medium=retriever,
    )
    evaluations = evaluate_experiment_groups(
        groups,
        [
            RetrievalCase(
                id="return",
                query="Как вернуть товар?",
                expected_chunk_ids=["returns-01"],
            )
        ],
    )
    experiments_path = tmp_path / "EXPERIMENTS.md"
    results_path = tmp_path / "retrieval-experiments.json"

    append_experiment_summary(
        evaluations,
        experiments_path,
        results_path,
    )

    report = experiments_path.read_text(encoding="utf-8")
    assert "Policy retrieval: chunk size and reranking" in report
    assert "| chunk_size | hybrid-small |" in report
    assert "| strategy | hybrid-reranked-medium |" in report
    assert str(results_path) in report
    assert "Decision:" in report