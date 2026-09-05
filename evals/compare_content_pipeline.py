import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marketplace_agent.config import Settings
from marketplace_agent.content.pipeline import run_content_pipeline
from marketplace_agent.evals.content_pipeline import (
    ContentPipelineCaseResult,
    ContentPipelineComparison,
    PipelineVersionMetrics,
    compare_pipeline_versions,
    load_content_pipeline_products,
    run_pipeline_version,
)
from marketplace_agent.evals.legacy_content_pipeline import (
    run_legacy_content_pipeline,
)
from marketplace_agent.llm.factory import create_llm_client


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare legacy and LangGraph content pipelines."
    )
    parser.add_argument("--legacy-version", required=True)
    parser.add_argument("--graph-version", required=True)
    parser.add_argument("--input-price-per-million", type=float)
    parser.add_argument("--output-price-per-million", type=float)
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser.parse_args()


def serialize_result(result: ContentPipelineCaseResult) -> dict[str, object]:
    return {
        "sku": result.sku,
        "status": result.status,
        "attempts": result.attempts,
        "true_attributes": dict(result.true_attributes),
        "extracted_attributes": dict(result.extracted_attributes),
        "used_attributes": dict(result.used_attributes),
        "violations": [
            violation.model_dump() for violation in result.violations
        ],
        "latency_ms": result.latency_ms,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "cost_usd": result.cost_usd,
        "model": result.model,
        "error": result.error,
        "trace": result.trace,
    }


def append_comparison_report(
    comparison: ContentPipelineComparison,
    legacy_version: str,
    graph_version: str,
    run_id: str,
    report_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("a", encoding="utf-8") as file:
        file.write(f"\n## Content pipeline comparison: {run_id}\n\n")
        file.write(
            "| Version | Success rate | Attribute F1 | Violations | "
            "Average latency, ms | Tokens | Cost, USD | Errors |\n"
        )
        file.write("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |\n")
        file.write(format_metrics_row(legacy_version, comparison.baseline))
        file.write(format_metrics_row(graph_version, comparison.validated))
        file.write(
            "\n"
            f"Исправленные SKU: {', '.join(comparison.corrected_skus) or 'нет'}.\n"
            f"Ухудшившиеся SKU: {', '.join(comparison.degraded_skus) or 'нет'}.\n"
        )


def format_metrics_row(
    version: str,
    metrics: PipelineVersionMetrics,
) -> str:
    token_count = metrics.prompt_tokens + metrics.completion_tokens
    failures = ", ".join(
        f"{error_type}={count}"
        for error_type, count in sorted(metrics.failures_by_type.items())
    ) or "нет"
    return (
        f"| {version} | {metrics.success_rate:.3f} | "
        f"{metrics.attribute_f1:.3f} | {metrics.violation_count} | "
        f"{metrics.average_latency_ms:.0f} | {token_count} | "
        f"{metrics.total_cost_usd:.6f} | {failures} |\n"
    )


def configured_model(settings: Settings) -> str:
    return {
        "groq": settings.groq_model,
        "openrouter": settings.openrouter_model,
        "gigachat": settings.gigachat_model,
    }[settings.llm_provider]


def print_progress(event: dict[str, object]) -> None:
    print(
        f"[{event['version']}] {event['position']}/{event['total']} "
        f"{event['sku']}: {event['event_type']}",
        flush=True,
    )


def main() -> None:
    arguments = parse_arguments()
    manifest_path = PROJECT_ROOT / "data/gold/content_pipeline_manifest.json"
    products = load_content_pipeline_products(manifest_path)
    settings = Settings.from_environment()
    input_price = (
        arguments.input_price_per_million
        if arguments.input_price_per_million is not None
        else settings.input_price_per_million
    )
    output_price = (
        arguments.output_price_per_million
        if arguments.output_price_per_million is not None
        else settings.output_price_per_million
    )
    progress = print_progress if os.getenv("LLM_PROGRESS") == "1" else None

    legacy_results = run_pipeline_version(
        products,
        llm_factory=lambda: create_llm_client(settings),
        pipeline=run_legacy_content_pipeline,
        max_attempts=arguments.max_attempts,
        input_price_per_million=input_price,
        output_price_per_million=output_price,
        version=arguments.legacy_version,
        progress=progress,
    )
    graph_results = run_pipeline_version(
        products,
        llm_factory=lambda: create_llm_client(settings),
        pipeline=run_content_pipeline,
        max_attempts=arguments.max_attempts,
        input_price_per_million=input_price,
        output_price_per_million=output_price,
        version=arguments.graph_version,
        progress=progress,
    )
    comparison = compare_pipeline_versions(legacy_results, graph_results)

    run_id = uuid4().hex
    output_path = PROJECT_ROOT / "evals/runs" / f"content-pipeline-{run_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "manifest_sha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
                "provider": settings.llm_provider,
                "model": configured_model(settings),
                "prices": {
                    "input_per_million": input_price,
                    "output_per_million": output_price,
                },
                "max_attempts": arguments.max_attempts,
                "legacy_version": arguments.legacy_version,
                "graph_version": arguments.graph_version,
                "legacy_results": [
                    serialize_result(result) for result in legacy_results
                ],
                "graph_results": [
                    serialize_result(result) for result in graph_results
                ],
                "comparison": {
                    "legacy": comparison.baseline.__dict__,
                    "graph": comparison.validated.__dict__,
                    "corrected_skus": comparison.corrected_skus,
                    "degraded_skus": comparison.degraded_skus,
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    append_comparison_report(
        comparison,
        arguments.legacy_version,
        arguments.graph_version,
        run_id,
        PROJECT_ROOT / "evals/REPORT.md",
    )

    print(f"Run ID: {run_id}")
    print(f"Legacy success rate: {comparison.baseline.success_rate:.3f}")
    print(f"Graph success rate: {comparison.validated.success_rate:.3f}")
    print(f"Results: {output_path}")


if __name__ == "__main__":
    main()
