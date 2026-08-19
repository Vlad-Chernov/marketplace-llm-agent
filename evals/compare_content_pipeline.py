import argparse
import json
import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marketplace_agent.config import Settings
from marketplace_agent.evals.content_pipeline import (
    ContentPipelineCaseResult,
    ContentPipelineComparison,
    PipelineVersionMetrics,
    compare_pipeline_versions,
    load_content_pipeline_products,
    run_pipeline_version,
)
from marketplace_agent.llm.factory import create_llm_client


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare baseline and validated content pipelines."
    )
    parser.add_argument("--version", required=True)
    parser.add_argument("--input-price-per-million", type=float, default=0.0)
    parser.add_argument("--output-price-per-million", type=float, default=0.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser.parse_args()


def serialize_result(result: ContentPipelineCaseResult) -> dict[str, object]:
    return {
        "sku": result.sku,
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
    }


def append_comparison_report(
    comparison: ContentPipelineComparison,
    version: str,
    run_id: str,
    report_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with report_path.open("a", encoding="utf-8") as file:
        file.write(f"\n## Content pipeline comparison: {run_id}\n\n")
        file.write(
            "| Version | Attribute F1 | Hallucination rate | Violations | "
            "Average latency, ms | Cost, USD |\n"
        )
        file.write("| --- | ---: | ---: | ---: | ---: | ---: |\n")
        file.write(
            format_metrics_row(
                "baseline-no-repair",
                comparison.baseline,
            )
        )
        file.write(format_metrics_row(version, comparison.validated))
        file.write(
            "\n"
            f"Исправленные SKU: {', '.join(comparison.corrected_skus) or 'нет'}.\n"
            f"Ухудшившиеся SKU: {', '.join(comparison.degraded_skus) or 'нет'}.\n"
        )


def format_metrics_row(
    version: str,
    metrics: PipelineVersionMetrics,
) -> str:
    return (
        f"| {version} | {metrics.attribute_f1:.3f} | "
        f"{metrics.hallucination_rate:.3f} | {metrics.violation_count} | "
        f"{metrics.average_latency_ms:.0f} | {metrics.total_cost_usd:.6f} |\n"
    )


def main() -> None:
    arguments = parse_arguments()
    manifest_path = (
        PROJECT_ROOT / "data" / "gold" / "content_pipeline_manifest.json"
    )
    products = load_content_pipeline_products(manifest_path)
    settings = Settings.from_environment()

    baseline_results = run_pipeline_version(
        products,
        create_llm_client(settings),
        max_attempts=1,
        input_price_per_million=arguments.input_price_per_million,
        output_price_per_million=arguments.output_price_per_million,
    )
    validated_results = run_pipeline_version(
        products,
        create_llm_client(settings),
        max_attempts=arguments.max_attempts,
        input_price_per_million=arguments.input_price_per_million,
        output_price_per_million=arguments.output_price_per_million,
    )
    comparison = compare_pipeline_versions(
        baseline_results,
        validated_results,
    )

    run_id = uuid4().hex
    output_path = PROJECT_ROOT / "evals" / "runs" / (
        f"content-pipeline-{run_id}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "version": arguments.version,
                "baseline": [
                    serialize_result(result) for result in baseline_results
                ],
                "validated": [
                    serialize_result(result) for result in validated_results
                ],
                "corrected_skus": comparison.corrected_skus,
                "degraded_skus": comparison.degraded_skus,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    append_comparison_report(
        comparison,
        arguments.version,
        run_id,
        PROJECT_ROOT / "evals" / "REPORT.md",
    )

    print(f"Run ID: {run_id}")
    print(f"Baseline F1: {comparison.baseline.attribute_f1:.3f}")
    print(f"Validated F1: {comparison.validated.attribute_f1:.3f}")
    print(f"Corrected SKU: {comparison.corrected_skus or 'нет'}")
    print(f"Degraded SKU: {comparison.degraded_skus or 'нет'}")
    print(f"Results: {output_path}")


if __name__ == "__main__":
    main()