from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from marketplace_agent.evals.retrieval_metrics import (
    RelevanceKey,
    RetrievalCase,
    RetrievalEvaluation,
    Retriever,
    evaluate_retriever,
)

ExperimentGroup = Literal["chunk_size", "strategy"]


@dataclass(frozen=True)
class RetrievalExperiment:
    """Describe one controlled retrieval evaluation."""

    name: str
    group: ExperimentGroup
    retriever: Retriever
    relevance_key: RelevanceKey


def build_experiment_groups(
    *,
    hybrid_small: Retriever,
    hybrid_medium: Retriever,
    hybrid_large: Retriever,
    vector_medium: Retriever,
    hybrid_reranked_medium: Retriever,
) -> dict[ExperimentGroup, list[RetrievalExperiment]]:
    """Build comparison groups that each vary one factor."""

    return {
        "chunk_size": [
            RetrievalExperiment(
                name="hybrid-small",
                group="chunk_size",
                retriever=hybrid_small,
                relevance_key="document_id",
            ),
            RetrievalExperiment(
                name="hybrid-medium",
                group="chunk_size",
                retriever=hybrid_medium,
                relevance_key="document_id",
            ),
            RetrievalExperiment(
                name="hybrid-large",
                group="chunk_size",
                retriever=hybrid_large,
                relevance_key="document_id",
            ),
        ],
        "strategy": [
            RetrievalExperiment(
                name="vector-medium",
                group="strategy",
                retriever=vector_medium,
                relevance_key="chunk_id",
            ),
            RetrievalExperiment(
                name="hybrid-medium",
                group="strategy",
                retriever=hybrid_medium,
                relevance_key="chunk_id",
            ),
            RetrievalExperiment(
                name="hybrid-reranked-medium",
                group="strategy",
                retriever=hybrid_reranked_medium,
                relevance_key="chunk_id",
            ),
        ],
    }

def evaluate_experiment_groups(
    groups: Mapping[ExperimentGroup, Sequence[RetrievalExperiment]],
    cases: Sequence[RetrievalCase],
    k: int = 5,
) -> dict[ExperimentGroup, dict[str, RetrievalEvaluation]]:
    """Evaluate every configuration without mixing controlled groups."""

    evaluations: dict[
        ExperimentGroup,
        dict[str, RetrievalEvaluation],
    ] = {}

    for group_name, experiments in groups.items():
        group_cases = (
            _with_medium_chunk_ids(cases)
            if group_name == "strategy"
            else cases
        )
        evaluations[group_name] = {
            experiment.name: evaluate_retriever(
                experiment.retriever,
                group_cases,
                k=k,
                relevance_key=experiment.relevance_key,
            )
            for experiment in experiments
        }

    return evaluations


def _with_medium_chunk_ids(
    cases: Sequence[RetrievalCase],
) -> list[RetrievalCase]:
    return [
        case.model_copy(
            update={
                "expected_chunk_ids": [
                    f"medium-{chunk_id}"
                    for chunk_id in case.expected_chunk_ids
                ]
            }
        )
        for case in cases
    ]

def append_experiment_summary(
    evaluations: Mapping[
        ExperimentGroup,
        Mapping[str, RetrievalEvaluation],
    ],
    experiments_path: Path,
    results_path: Path,
) -> None:
    """Append measured experiment results and a deterministic decision."""

    experiments_path.parent.mkdir(parents=True, exist_ok=True)
    winner_name, _ = max(
        evaluations["strategy"].items(),
        key=lambda item: (
            item[1].recall_at_k,
            item[1].mrr,
            -item[1].mean_latency_ms,
        ),
    )

    with experiments_path.open("a", encoding="utf-8") as experiments_file:
        experiments_file.write(
            "\n## 2026-08-21 — Policy retrieval: chunk size and reranking\n\n"
        )
        experiments_file.write(
            "| Group | Configuration | Recall@5 | MRR | "
            "Mean latency, ms |\n"
        )
        experiments_file.write(
            "| --- | --- | ---: | ---: | ---: |\n"
        )

        for group_name, group_evaluations in evaluations.items():
            for name, evaluation in group_evaluations.items():
                experiments_file.write(
                    f"| {group_name} | {name} "
                    f"| {evaluation.recall_at_k:.3f} "
                    f"| {evaluation.mrr:.3f} "
                    f"| {evaluation.mean_latency_ms:.3f} |\n"
                )

        experiments_file.write(f"\nResult JSON: `{results_path}`.\n\n")
        experiments_file.write(
            f"Decision: use `{winner_name}` because it has the best "
            "strategy score by Recall@5, MRR, then mean latency.\n"
        )