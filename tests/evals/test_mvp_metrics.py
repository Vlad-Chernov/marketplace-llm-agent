import pytest

from marketplace_agent.evals.mvp_metrics import summarize_mvp_run
from marketplace_agent.evals.runner import (
    EvaluationResult,
    EvaluationRun,
)


def test_summarizes_mvp_run_quality_and_resources() -> None:
    run = EvaluationRun(
        run_id="run-001",
        suite="mvp-final",
        version="MVP-final",
        created_at="2026-08-22T12:00:00+00:00",
        results=[
            EvaluationResult(
                case_id="gold-001",
                case_type="attribute_extraction",
                prediction={"ram_gb": "16"},
                raw_response='{"ram_gb":"16"}',
                model="fake-model",
                reference={"ram_gb": "16"},
                error=None,
                latency_ms=10,
                prompt_tokens=10,
                completion_tokens=5,
                cost_usd=0.00002,
            ),
            EvaluationResult(
                case_id="gold-009",
                case_type="validation",
                prediction={"violations": ["uppercase-ratio"]},
                raw_response='{"violations":["uppercase-ratio"]}',
                model="fake-model",
                reference={"violations": ["uppercase-ratio"]},
                error=None,
                latency_ms=20,
                prompt_tokens=20,
                completion_tokens=10,
                cost_usd=0.00004,
            ),
            EvaluationResult(
                case_id="gold-023",
                case_type="review_analysis",
                prediction={"defect_label": "battery_drain"},
                raw_response='{"defect_label":"battery_drain"}',
                model="fake-model",
                reference={"defect_label": "battery_drain"},
                error=None,
                latency_ms=50,
                prompt_tokens=30,
                completion_tokens=15,
                cost_usd=0.00006,
            ),
        ],
    )

    summary = summarize_mvp_run(run)

    assert summary.case_count == 3
    assert summary.error_count == 0
    assert summary.attribute_f1 == pytest.approx(1.0)
    assert summary.validation_accuracy == pytest.approx(1.0)
    assert summary.review_recall == pytest.approx(1.0)
    assert summary.total_prompt_tokens == 60
    assert summary.total_completion_tokens == 30
    assert summary.total_cost_usd == pytest.approx(0.00012)
    assert summary.latency_p50_ms == 20
    assert summary.latency_p95_ms == 50