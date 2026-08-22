from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from typing import Any

from marketplace_agent.evals.attribute_metrics import (
    evaluate_attribute_extraction,
)
from marketplace_agent.evals.runner import EvaluationRun


@dataclass(frozen=True)
class MvpRunSummary:
    """Ключевые метрики одного прогона MVP."""

    case_count: int
    error_count: int
    attribute_f1: float
    validation_accuracy: float
    review_recall: float
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cost_usd: float
    latency_p50_ms: int
    latency_p95_ms: int


def summarize_mvp_run(run: EvaluationRun) -> MvpRunSummary:
    """Посчитать качество и ресурсы по результатам прогона."""

    successful_results = [
        result for result in run.results if result.error is None
    ]
    attribute_results = [
        result
        for result in successful_results
        if result.case_type == "attribute_extraction"
    ]
    attribute_metrics = evaluate_attribute_extraction(
        [
            _as_mapping(result.prediction)
            for result in attribute_results
        ],
        [
            _as_mapping(result.reference)
            for result in attribute_results
        ],
    )

    validation_results = [
        result
        for result in successful_results
        if result.case_type == "validation"
    ]
    review_results = [
        result
        for result in successful_results
        if result.case_type == "review_analysis"
    ]

    latencies = [result.latency_ms for result in run.results]

    return MvpRunSummary(
        case_count=len(run.results),
        error_count=sum(result.error is not None for result in run.results),
        attribute_f1=attribute_metrics.overall.f1,
        validation_accuracy=_validation_accuracy(validation_results),
        review_recall=_review_recall(review_results),
        total_prompt_tokens=sum(
            result.prompt_tokens for result in run.results
        ),
        total_completion_tokens=sum(
            result.completion_tokens for result in run.results
        ),
        total_cost_usd=sum(result.cost_usd for result in run.results),
        latency_p50_ms=_percentile(latencies, 50),
        latency_p95_ms=_percentile(latencies, 95),
    )


def _validation_accuracy(results: Sequence[Any]) -> float:
    if not results:
        return 0.0

    correct_count = sum(
        set(_as_mapping(result.prediction).get("violations", []))
        == set(_as_mapping(result.reference).get("violations", []))
        for result in results
    )
    return correct_count / len(results)


def _review_recall(results: Sequence[Any]) -> float:
    if not results:
        return 0.0

    correct_count = sum(
        _as_mapping(result.prediction).get("defect_label")
        == _as_mapping(result.reference).get("defect_label")
        for result in results
    )
    return correct_count / len(results)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("Evaluation prediction must be a mapping.")
    return value


def _percentile(values: Sequence[int], percentile: int) -> int:
    if not values:
        return 0

    sorted_values = sorted(values)
    index = ceil(percentile / 100 * len(sorted_values)) - 1
    return sorted_values[index]