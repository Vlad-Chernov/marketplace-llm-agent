import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

from marketplace_agent.evals.models import GoldenCase


@dataclass(frozen=True)
class EvaluationPrediction:
    prediction: Any
    raw_response: str
    model: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    tool_calls: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvaluationResult:
    case_id: str
    case_type: str
    prediction: Any | None
    raw_response: str | None
    model: str | None
    reference: Any
    error: str | None
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float
    tool_calls: list[str] = field(default_factory=list)
    expected_tool_calls: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class EvaluationRun:
    run_id: str
    suite: str
    version: str
    created_at: str
    results: list[EvaluationResult]


@dataclass(frozen=True)
class EvaluationProgress:
    """Progress information emitted after a completed evaluation case."""

    current: int
    total: int
    case_id: str
    case_type: str
    elapsed_seconds: float
    estimated_remaining_seconds: float


def run_evaluation(
    cases: Sequence[GoldenCase],
    suite: str,
    version: str,
    execute_case: Callable[[GoldenCase], EvaluationPrediction],
    progress_callback: Callable[[EvaluationProgress], None] | None = None,
) -> EvaluationRun:
    results: list[EvaluationResult] = []

    total_cases = len(cases)
    run_started_at = perf_counter()

    for index, case in enumerate(cases, start=1):
        started_at = perf_counter()

        try:
            prediction = execute_case(case)
        except Exception as error:  # noqa: BLE001
            latency_ms = round((perf_counter() - started_at) * 1000)
            results.append(
                EvaluationResult(
                    case_id=case.id,
                    case_type=case.type,
                    prediction=None,
                    raw_response=None,
                    model=None,
                    reference=case.expected_answer,
                    error=f"{type(error).__name__}: {error}",
                    latency_ms=latency_ms,
                    prompt_tokens=0,
                    completion_tokens=0,
                    cost_usd=0.0,
                    expected_tool_calls=case.must_call_tools,
                )
            )
        else:
            results.append(
                EvaluationResult(
                    case_id=case.id,
                    case_type=case.type,
                    prediction=prediction.prediction,
                    raw_response=prediction.raw_response,
                    model=prediction.model,
                    reference=case.expected_answer,
                    error=None,
                    latency_ms=prediction.latency_ms,
                    prompt_tokens=prediction.prompt_tokens,
                    completion_tokens=prediction.completion_tokens,
                    cost_usd=prediction.cost_usd,
                    tool_calls=prediction.tool_calls,
                    expected_tool_calls=case.must_call_tools,
                )
            )

        if progress_callback is not None:
            elapsed_seconds = perf_counter() - run_started_at
            average_seconds = elapsed_seconds / index
            progress_callback(
                EvaluationProgress(
                    current=index,
                    total=total_cases,
                    case_id=case.id,
                    case_type=case.type,
                    elapsed_seconds=elapsed_seconds,
                    estimated_remaining_seconds=(
                        average_seconds * (total_cases - index)
                    ),
                )
            )

    return EvaluationRun(
        run_id=uuid4().hex,
        suite=suite,
        version=version,
        created_at=datetime.now(UTC).isoformat(),
        results=results,
    )

def load_evaluation_run(path: Path) -> EvaluationRun:
    """Загрузить ранее сохранённый результат evaluation-прогона."""

    payload = json.loads(path.read_text(encoding="utf-8"))

    return EvaluationRun(
        run_id=payload["run_id"],
        suite=payload["suite"],
        version=payload["version"],
        created_at=payload["created_at"],
        results=[
            EvaluationResult(**result)
            for result in payload["results"]
        ],
    )

def save_run(run: EvaluationRun, output_directory: Path) -> Path:
    output_directory.mkdir(parents=True, exist_ok=True)
    file_name = f"{run.run_id}-{uuid4().hex[:8]}.json"
    output_path = output_directory / file_name

    payload = {
        "run_id": run.run_id,
        "suite": run.suite,
        "version": run.version,
        "created_at": run.created_at,
        "results": [asdict(result) for result in run.results],
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return output_path

def append_report_summary(
    run: EvaluationRun,
    report_path: Path,
    summary_metrics: Mapping[str, float] | None = None,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    header = (
        "\n## Run history\n\n"
        "| Run ID | Suite | Version | Cases | Errors | Average latency, ms | "
        "Cost, USD | Metrics |\n"
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |\n"
    )

    if report_path.exists():
        report_content = report_path.read_text(encoding="utf-8")
    else:
        report_content = "# Evaluation Report\n"

    if "## Run history" not in report_content:
        with report_path.open("a", encoding="utf-8") as file:
            file.write(header)

    case_count = len(run.results)
    error_count = sum(result.error is not None for result in run.results)
    average_latency_ms = (
        round(sum(result.latency_ms for result in run.results) / case_count)
        if case_count
        else 0
    )
    total_cost_usd = sum(result.cost_usd for result in run.results)
    metrics_text = (
        ", ".join(
            f"{name}={value:.3f}" for name, value in summary_metrics.items()
        )
        if summary_metrics
        else "-"
    )

    with report_path.open("a", encoding="utf-8") as file:
        file.write(
            f"| {run.run_id} | {run.suite} | {run.version} | "
            f"{case_count} | {error_count} | {average_latency_ms} | "
            f"{total_cost_usd:.6f} | {metrics_text} |\n"
        )
