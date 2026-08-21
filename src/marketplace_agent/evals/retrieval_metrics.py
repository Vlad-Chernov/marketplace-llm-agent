from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from marketplace_agent.retrieval.lexical import SearchResult

RelevanceKey = Literal["chunk_id", "document_id"]


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
    relevance_key: RelevanceKey
    expected_chunk_ids: list[str]
    expected_ids: list[str]
    retrieved_chunk_ids: list[str]
    retrieved_document_ids: list[str]
    recall_at_k: float
    first_relevant_rank: int | None
    reciprocal_rank: float
    latency_ms: float


class RetrievalEvaluation(BaseModel):
    """Store aggregate retrieval metrics and per-case results."""

    model_config = ConfigDict(extra="forbid")

    recall_at_k: float
    mrr: float
    mean_latency_ms: float
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
    relevance_key: RelevanceKey = "chunk_id",
) -> RetrievalEvaluation:
    """Measure Recall@k, MRR, and search latency for one retriever."""

    results: list[RetrievalCaseResult] = []

    for case in cases:
        started_at = perf_counter()
        search_results = retriever.search(case.query, k, case.filters)
        latency_ms = round((perf_counter() - started_at) * 1000, 3)

        retrieved_chunk_ids = [
            result.chunk.chunk_id for result in search_results[:k]
        ]
        retrieved_document_ids = [
            result.chunk.document_id for result in search_results[:k]
        ]
        expected_ids = _expected_ids(case, relevance_key)
        retrieved_ids = (
            retrieved_chunk_ids
            if relevance_key == "chunk_id"
            else retrieved_document_ids
        )
        expected_id_set = set(expected_ids)
        first_relevant_rank = next(
            (
                rank
                for rank, identifier in enumerate(retrieved_ids, start=1)
                if identifier in expected_id_set
            ),
            None,
        )
        recall_at_k = len(expected_id_set.intersection(retrieved_ids)) / len(
            expected_id_set
        )
        reciprocal_rank = (
            0.0
            if first_relevant_rank is None
            else 1 / first_relevant_rank
        )

        results.append(
            RetrievalCaseResult(
                case_id=case.id,
                relevance_key=relevance_key,
                expected_chunk_ids=case.expected_chunk_ids,
                expected_ids=expected_ids,
                retrieved_chunk_ids=retrieved_chunk_ids,
                retrieved_document_ids=retrieved_document_ids,
                recall_at_k=recall_at_k,
                first_relevant_rank=first_relevant_rank,
                reciprocal_rank=reciprocal_rank,
                latency_ms=latency_ms,
            )
        )

    return RetrievalEvaluation(
        recall_at_k=_mean(result.recall_at_k for result in results),
        mrr=_mean(result.reciprocal_rank for result in results),
        mean_latency_ms=_mean(result.latency_ms for result in results),
        results=results,
    )


def append_retrieval_report(
    evaluations: Mapping[str, RetrievalEvaluation],
    report_path: Path,
    k: int,
) -> None:
    """Append retrieval metrics and missed chunks to Markdown."""

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("a", encoding="utf-8") as report_file:
        report_file.write(f"\n## Policy retrieval comparison (Recall@{k})\n\n")
        report_file.write("| Retriever | Recall@k | MRR | Mean latency, ms |\n")
        report_file.write("| --- | ---: | ---: | ---: |\n")
        for name, evaluation in evaluations.items():
            report_file.write(
                f"| {name} | {evaluation.recall_at_k:.3f} "
                f"| {evaluation.mrr:.3f} "
                f"| {evaluation.mean_latency_ms:.3f} |\n"
            )

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


def _expected_ids(
    case: RetrievalCase,
    relevance_key: RelevanceKey,
) -> list[str]:
    if relevance_key == "chunk_id":
        return case.expected_chunk_ids

    return [
        chunk_id.rsplit("-", maxsplit=1)[0]
        for chunk_id in case.expected_chunk_ids
    ]


def _mean(values: Sequence[float] | object) -> float:
    value_list = list(values)
    return sum(value_list) / len(value_list) if value_list else 0.0