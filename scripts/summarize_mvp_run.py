import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marketplace_agent.evals.mvp_metrics import summarize_mvp_run
from marketplace_agent.evals.runner import load_evaluation_run


def main() -> None:
    """Print summary metrics for one saved MVP evaluation run."""

    result_path = Path(sys.argv[1])
    run = load_evaluation_run(result_path)
    summary = summarize_mvp_run(run)

    print(f"Cases: {summary.case_count}")
    print(f"Errors: {summary.error_count}")
    print(f"Attribute F1: {summary.attribute_f1:.3f}")
    print(f"Validation accuracy: {summary.validation_accuracy:.3f}")
    print(f"Review recall: {summary.review_recall:.3f}")
    print("Support tool accuracy: "
        f"{summary.support_tool_accuracy:.3f}"
    )
    print(f"Prompt tokens: {summary.total_prompt_tokens}")
    print(f"Completion tokens: {summary.total_completion_tokens}")
    print(f"Cost, USD: {summary.total_cost_usd:.6f}")
    print(f"Latency p50, ms: {summary.latency_p50_ms}")
    print(f"Latency p95, ms: {summary.latency_p95_ms}")


if __name__ == "__main__":
    main()