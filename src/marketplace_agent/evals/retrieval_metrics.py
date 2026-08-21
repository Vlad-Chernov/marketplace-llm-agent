from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from marketplace_agent.retrieval.lexical import SearchResult


class RetrievalCase(BaseModel):
    """Store one manually reviewed retrieval task."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected_chunk_ids: list[str] = Field(min_length=1)
    filters: dict[str, str] | None = None


class RetrievalCaseResult(BaseModel):
    """Store one retriever result for one evaluation case."""

    model_config = ConfigDict(extra="forbid")

    case_id: str
    expected_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    recall_at_k: float


class RetrievalEvaluation(BaseModel):
    """Store aggregate retrieval metrics and per-case results."""

    model_config = ConfigDict(extra="forbid")

    recall_at_k: float
    results: list[RetrievalCaseResult]


class Retriever(Protocol):
    """Search policy chunks."""

    def search(
        self,
        query: str,
        k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[SearchResult]: ...


def evaluate_retriever(
    retriever: Retriever,
    cases: Sequence[RetrievalCase],
    k: int = 5,
) -> RetrievalEvaluation:
    """Measure mean Recall@k for one retriever."""

    results: list[RetrievalCaseResult] = []

    for case in cases:
        search_results = retriever.search(case.query, k, case.filters)
        retrieved_chunk_ids = [
            result.chunk.chunk_id for result in search_results[:k]
        ]
        expected_chunk_ids = set(case.expected_chunk_ids)
        recall_at_k = len(
            expected_chunk_ids.intersection(retrieved_chunk_ids)
        ) / len(expected_chunk_ids)

        results.append(
            RetrievalCaseResult(
                case_id=case.id,
                expected_chunk_ids=case.expected_chunk_ids,
                retrieved_chunk_ids=retrieved_chunk_ids,
                recall_at_k=recall_at_k,
            )
        )

    mean_recall = (
        sum(result.recall_at_k for result in results) / len(results)
        if results
        else 0.0
    )
    return RetrievalEvaluation(recall_at_k=mean_recall, results=results)


def append_retrieval_report(
    evaluations: Mapping[str, RetrievalEvaluation],
    report_path: Path,
    k: int,
) -> None:
    """Append retrieval metrics and missed chunks to Markdown."""

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("a", encoding="utf-8") as report_file:
        report_file.write(f"\n## Policy retrieval comparison (Recall@{k})\n\n")
        report_file.write("| Retriever | Recall@k |\n")
        report_file.write("| --- | --- |\n")
        for name, evaluation in evaluations.items():
            report_file.write(f"| {name} | {evaluation.recall_at_k:.3f} |\n")

        report_file.write("\n### Missing chunks\n\n")
        for name, evaluation in evaluations.items():
            for result in evaluation.results:
                missing_chunk_ids = sorted(
                    set(result.expected_chunk_ids)
                    - set(result.retrieved_chunk_ids)
                )
                if missing_chunk_ids:
                    report_file.write(
                        f"- {name}, {result.case_id}: "
                        f"{', '.join(missing_chunk_ids)}\n"
                    )