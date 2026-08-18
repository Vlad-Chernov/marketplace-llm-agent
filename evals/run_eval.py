import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marketplace_agent.config import Settings
from marketplace_agent.evals.attribute_metrics import (
    evaluate_attribute_extraction,
)
from marketplace_agent.evals.baseline import create_case_executor
from marketplace_agent.evals.models import GoldenCase
from marketplace_agent.evals.runner import (
    EvaluationRun,
    append_report_summary,
    run_evaluation,
    save_run,
)
from marketplace_agent.llm.factory import create_llm_client


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a reproducible evaluation suite."
    )
    parser.add_argument("--suite", required=True, choices=["mvp"])
    parser.add_argument("--version", required=True)
    parser.add_argument("--case-type")
    parser.add_argument("--input-price-per-million", type=float, default=0.0)
    parser.add_argument("--output-price-per-million", type=float, default=0.0)
    return parser.parse_args()


def load_cases(suite: str) -> list[GoldenCase]:
    cases_path = PROJECT_ROOT / "data" / "gold" / f"{suite}_cases.json"
    raw_cases = json.loads(cases_path.read_text(encoding="utf-8"))
    return [GoldenCase.model_validate(raw_case) for raw_case in raw_cases]



def calculate_attribute_metrics(run: EvaluationRun) -> dict[str, float]:
    attribute_results = [
        result
        for result in run.results
        if result.case_type == "attribute_extraction"
    ]
    if not attribute_results:
        return {}

    predictions: list[Mapping[str, Any]] = []
    references: list[Mapping[str, Any]] = []

    for result in attribute_results:
        prediction = (
            result.prediction
            if isinstance(result.prediction, Mapping)
            else {}
        )
        reference = (
            result.reference
            if isinstance(result.reference, Mapping)
            else {}
        )
        predictions.append(prediction)
        references.append(reference)

    metrics = evaluate_attribute_extraction(predictions, references)
    return {
        "attribute_f1": metrics.overall.f1,
        "hallucination_rate": metrics.hallucination_rate,
    }

def main() -> None:
    arguments = parse_arguments()
    cases = load_cases(arguments.suite)

    if arguments.case_type is not None:
        cases = [case for case in cases if case.type == arguments.case_type]

    if not cases:
        raise SystemExit("No cases match the selected suite and type.")

    client = create_llm_client(Settings.from_environment())
    run = run_evaluation(
        cases=cases,
        suite=arguments.suite,
        version=arguments.version,
        execute_case=create_case_executor(
            client,
            arguments.input_price_per_million,
            arguments.output_price_per_million,
        ),
    )

    results_path = save_run(run, PROJECT_ROOT / "evals" / "runs")
    summary_metrics = calculate_attribute_metrics(run)
    append_report_summary(
        run,
        PROJECT_ROOT / "evals" / "REPORT.md",
        summary_metrics=summary_metrics,
    )

    error_count = sum(result.error is not None for result in run.results)
    print(f"Run ID: {run.run_id}")
    print(f"Cases: {len(run.results)}")
    print(f"Errors: {error_count}")
    for name, value in summary_metrics.items():
        print(f"{name}: {value:.3f}")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()