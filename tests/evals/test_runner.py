from pathlib import Path

from marketplace_agent.evals.models import GoldenCase
from marketplace_agent.evals.runner import (
    EvaluationPrediction,
    append_report_summary,
    load_evaluation_run,
    run_evaluation,
    save_run,
)


def test_runs_case_and_persists_non_overwriting_results(tmp_path: Path) -> None:
    case = GoldenCase(
        id="gold-001",
        question="Извлеки ОЗУ: 16 ГБ.",
        type="attribute_extraction",
        input={"supplier_description": "Ноутбук с ОЗУ 16 ГБ."},
        expected_answer={"ram_gb": "16"},
        origin="manual",
    )

    def execute_case(_: GoldenCase) -> EvaluationPrediction:
        return EvaluationPrediction(
            prediction={"ram_gb": "16"},
            raw_response='{"ram_gb": "16"}',
            model="fake-model-v1",
            latency_ms=25,
            prompt_tokens=10,
            completion_tokens=5,
            cost_usd=0.00001,
        )

    run = run_evaluation(
        cases=[case],
        suite="mvp",
        version="baseline",
        execute_case=execute_case,
    )

    assert run.suite == "mvp"
    assert run.version == "baseline"
    assert run.results[0].case_id == "gold-001"
    assert run.results[0].prediction == {"ram_gb": "16"}
    assert run.results[0].reference == {"ram_gb": "16"}
    assert run.results[0].model == "fake-model-v1"
    assert run.results[0].error is None

    first_path = save_run(run, tmp_path)
    second_path = save_run(run, tmp_path)

    assert first_path != second_path
    assert first_path.exists()
    assert second_path.exists()

    report_path = tmp_path / "REPORT.md"
    append_report_summary(run, report_path)
    append_report_summary(
    run,
    report_path,
    summary_metrics={"attribute_f1": 0.5},
    )

    assert report_path.read_text(encoding="utf-8").count(run.run_id) == 2
    assert "attribute_f1=0.500" in report_path.read_text(encoding="utf-8")

def test_loads_saved_evaluation_run(tmp_path: Path) -> None:
    case = GoldenCase(
        id="gold-001",
        question="Извлеки ОЗУ.",
        type="attribute_extraction",
        input={"supplier_description": "Ноутбук с ОЗУ 16 ГБ."},
        expected_answer={"ram_gb": "16"},
        origin="manual",
    )
    run = run_evaluation(
        cases=[case],
        suite="mvp-final",
        version="MVP-final",
        execute_case=lambda _: EvaluationPrediction(
            prediction={"ram_gb": "16"},
            raw_response='{"ram_gb":"16"}',
            model="fake-model",
            latency_ms=12,
            prompt_tokens=10,
            completion_tokens=5,
            cost_usd=0.00002,
        ),
    )
    path = save_run(run, tmp_path)

    loaded_run = load_evaluation_run(path)

    assert loaded_run.run_id == run.run_id
    assert loaded_run.suite == "mvp-final"
    assert loaded_run.results[0] == run.results[0]