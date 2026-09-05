from pathlib import Path

from marketplace_agent.evals.models import GoldenCase
from marketplace_agent.evals.runner import (
    EvaluationPrediction,
    EvaluationProgress,
    EvaluationResult,
    EvaluationRun,
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
            tool_calls=["search_policy"],
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
    assert run.results[0].tool_calls == ["search_policy"]
    assert run.results[0].expected_tool_calls == []
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
            tool_calls=["search_policy"],
        ),
    )
    path = save_run(run, tmp_path)

    loaded_run = load_evaluation_run(path)

    assert loaded_run.run_id == run.run_id
    assert loaded_run.suite == "mvp-final"
    assert loaded_run.results[0] == run.results[0]


def test_reports_completed_case_with_estimated_remaining_time(
    monkeypatch,
) -> None:
    cases = [
        GoldenCase(
            id=f"gold-00{index}",
            question="Вопрос",
            type="attribute_extraction",
            input={"supplier_description": "Описание"},
            expected_answer={},
            origin="manual",
        )
        for index in range(1, 3)
    ]
    progress_events: list[EvaluationProgress] = []
    clock_values = iter([0.0, 0.0, 2.0, 2.0, 5.0])
    monkeypatch.setattr(
        "marketplace_agent.evals.runner.perf_counter",
        lambda: next(clock_values),
    )

    run_evaluation(
        cases=cases,
        suite="mvp",
        version="baseline",
        execute_case=lambda _: EvaluationPrediction(
            prediction={},
            raw_response="{}",
            model="fake-model",
            latency_ms=0,
            prompt_tokens=0,
            completion_tokens=0,
            cost_usd=0.0,
        ),
        progress_callback=progress_events.append,
    )

    assert progress_events == [
        EvaluationProgress(
            current=1,
            total=2,
            case_id="gold-001",
            case_type="attribute_extraction",
            elapsed_seconds=2.0,
            estimated_remaining_seconds=2.0,
        ),
        EvaluationProgress(
            current=2,
            total=2,
            case_id="gold-002",
            case_type="attribute_extraction",
            elapsed_seconds=5.0,
            estimated_remaining_seconds=0.0,
        ),
    ]
def test_save_run_redacts_pii_from_raw_response_and_error(
    tmp_path: Path,
) -> None:
    run = EvaluationRun(
        run_id="run-pii",
        suite="mvp",
        version="test",
        created_at="2026-09-05T00:00:00+00:00",
        results=[
            EvaluationResult(
                case_id="gold-pii-001",
                case_type="support",
                prediction={},
                raw_response=(
                    "Анна Петрова, телефон +7 900 111-22-33."
                ),
                model="fake-model",
                reference={},
                error=None,
                latency_ms=0,
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd=0.0,
            ),
            EvaluationResult(
                case_id="gold-pii-002",
                case_type="support",
                prediction=None,
                raw_response=None,
                model=None,
                reference={},
                error=(
                    "Ошибка для anna@example.com: "
                    "улица Ленина, дом 10."
                ),
                latency_ms=0,
                prompt_tokens=0,
                completion_tokens=0,
                cost_usd=0.0,
            ),
        ],
    )

    output_path = save_run(run, tmp_path)
    saved_text = output_path.read_text(encoding="utf-8")

    assert "Анна Петрова" not in saved_text
    assert "+7 900 111-22-33" not in saved_text
    assert "anna@example.com" not in saved_text
    assert "улица Ленина, дом 10" not in saved_text
    assert "[PERSON]" in saved_text
    assert "[PHONE]" in saved_text
    assert "[EMAIL]" in saved_text
    assert "[ADDRESS]" in saved_text