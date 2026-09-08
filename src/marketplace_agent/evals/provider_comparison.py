import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from marketplace_agent.evals.attribute_metrics import evaluate_attribute_extraction
from marketplace_agent.evals.llm_meter import MeteredLLMClient
from marketplace_agent.evals.models import GoldenCase
from marketplace_agent.evals.mvp_executor import MvpCaseExecutor
from marketplace_agent.evals.mvp_metrics import summarize_mvp_run
from marketplace_agent.evals.runner import (
    EvaluationPrediction,
    EvaluationRun,
    run_evaluation,
)
from marketplace_agent.llm.base import LLMClient


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    model: str
    api_key: str
    client_factory: Callable[[], LLMClient]
    executor_factory: Callable[[MeteredLLMClient], MvpCaseExecutor]
    input_price_per_million: float = 0.0
    output_price_per_million: float = 0.0


@dataclass(frozen=True)
class ProviderRunSummary:
    provider: str
    model: str
    run_id: str
    case_count: int
    error_count: int
    attribute_f1: float
    hallucination_rate: float
    validation_accuracy: float
    review_recall: float
    support_accuracy: float
    latency_p50_ms: int
    latency_p95_ms: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    cost_status: str


@dataclass(frozen=True)
class ProviderComparisonResult:
    created_at: str
    case_count: int
    dataset_sha256: str
    runs: tuple[ProviderRunSummary, ...]


def run_provider_comparison(
    cases: Sequence[GoldenCase],
    providers: Sequence[ProviderSpec],
    progress_callback: Callable[[str, str, int, int, str], None] | None = None,
) -> ProviderComparisonResult:
    case_list = list(cases)
    summaries: list[ProviderRunSummary] = []

    for provider in providers:
        run = _run_one_provider(
            provider,
            case_list,
            progress_callback,
        )
        summaries.append(_summarize(provider, run))

    return ProviderComparisonResult(
        created_at=datetime.now(UTC).isoformat(),
        case_count=len(case_list),
        dataset_sha256=_dataset_hash(case_list),
        runs=tuple(summaries),
    )


def _run_one_provider(
    provider: ProviderSpec,
    cases: Sequence[GoldenCase],
    progress_callback: Callable[[str, str, int, int, str], None] | None,
) -> EvaluationRun:
    def execute(case: GoldenCase) -> EvaluationPrediction:
        meter = MeteredLLMClient(
            provider.client_factory(),
            input_price_per_million=provider.input_price_per_million,
            output_price_per_million=provider.output_price_per_million,
        )
        return provider.executor_factory(meter).execute(case)

    def report(progress: Any) -> None:
        if progress_callback is not None:
            progress_callback(
                provider.name,
                progress.case_id,
                progress.current,
                progress.total,
                "completed",
            )

    return run_evaluation(
        cases,
        suite="mvp-provider-comparison",
        version=provider.model,
        execute_case=execute,
        progress_callback=report,
    )


def serialize_provider_comparison(
    result: ProviderComparisonResult,
) -> dict[str, object]:
    return {
        "created_at": result.created_at,
        "case_count": result.case_count,
        "dataset_sha256": result.dataset_sha256,
        "runs": [asdict(run) for run in result.runs],
    }


def _summarize(provider: ProviderSpec, run: EvaluationRun) -> ProviderRunSummary:
    metrics = summarize_mvp_run(run)
    attribute_results = [
        result
        for result in run.results
        if result.case_type == "attribute_extraction" and result.error is None
    ]
    attribute_metrics = evaluate_attribute_extraction(
        [result.prediction for result in attribute_results],
        [result.reference for result in attribute_results],
    )
    return ProviderRunSummary(
        provider=provider.name,
        model=provider.model,
        run_id=run.run_id,
        case_count=metrics.case_count,
        error_count=metrics.error_count,
        attribute_f1=metrics.attribute_f1,
        hallucination_rate=attribute_metrics.hallucination_rate,
        validation_accuracy=metrics.validation_accuracy,
        review_recall=metrics.review_recall,
        support_accuracy=metrics.support_tool_accuracy,
        latency_p50_ms=metrics.latency_p50_ms,
        latency_p95_ms=metrics.latency_p95_ms,
        prompt_tokens=metrics.total_prompt_tokens,
        completion_tokens=metrics.total_completion_tokens,
        cost_usd=metrics.total_cost_usd,
        cost_status=(
            "unmeasured"
            if provider.input_price_per_million == 0
            and provider.output_price_per_million == 0
            else "measured"
        ),
    )


def _dataset_hash(cases: Sequence[GoldenCase]) -> str:
    payload = json.dumps(
        [case.model_dump(mode="json") for case in cases],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
