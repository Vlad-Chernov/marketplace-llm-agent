import json
from pathlib import Path

from marketplace_agent.evals.models import GoldenCase
from marketplace_agent.evals.mvp_final import (
    load_golden_cases,
    run_mvp_final,
)
from marketplace_agent.evals.runner import EvaluationPrediction


def test_loads_cases_and_runs_mvp_evaluation(tmp_path: Path) -> None:
    cases_path = tmp_path / "cases.json"
    cases_path.write_text(
        json.dumps(
            [
                {
                    "id": "gold-001",
                    "question": "Извлеки ОЗУ.",
                    "type": "attribute_extraction",
                    "input": {
                        "supplier_description": "Ноутбук с ОЗУ 16 ГБ."
                    },
                    "expected_answer": {"ram_gb": "16"},
                    "origin": "manual",
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cases = load_golden_cases(cases_path)

    def execute_case(case: GoldenCase) -> EvaluationPrediction:
        assert case.id == "gold-001"
        return EvaluationPrediction(
            prediction={"ram_gb": "16"},
            raw_response='{"ram_gb":"16"}',
            model="fake-model",
            latency_ms=12,
            prompt_tokens=10,
            completion_tokens=5,
            cost_usd=0.00002,
        )

    run = run_mvp_final(cases, execute_case)

    assert run.suite == "mvp-final"
    assert run.version == "MVP-final"
    assert run.results[0].prediction == {"ram_gb": "16"}
    assert run.results[0].error is None