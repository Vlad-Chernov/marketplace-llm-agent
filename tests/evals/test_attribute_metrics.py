import pytest

from marketplace_agent.evals.attribute_metrics import (
    evaluate_attribute_extraction,
)


def test_calculates_attribute_metrics_and_error_groups() -> None:
    predictions = [
        {
            "ram_gb": "16",
            "screen_diagonal_in": 15.6,
            "color": "black",
            "battery_capacity_wh": None,
        }
    ]
    references = [
        {
            "ram_gb": "16",
            "screen_diagonal_in": 14.0,
            "color": None,
            "battery_capacity_wh": 50,
        }
    ]

    metrics = evaluate_attribute_extraction(predictions, references)

    assert metrics.overall.true_positive == 1
    assert metrics.overall.false_positive == 2
    assert metrics.overall.false_negative == 2
    assert metrics.overall.precision == pytest.approx(1 / 3)
    assert metrics.overall.recall == pytest.approx(1 / 3)
    assert metrics.overall.f1 == pytest.approx(1 / 3)

    assert metrics.hallucination_count == 1
    assert metrics.hallucination_rate == pytest.approx(1 / 3)

    assert metrics.required.true_positive == 1
    assert metrics.required.false_positive == 1
    assert metrics.required.false_negative == 1

    assert metrics.filterable.true_positive == 1
    assert metrics.filterable.false_positive == 2
    assert metrics.filterable.false_negative == 1


def test_returns_perfect_score_for_exact_prediction() -> None:
    predictions = [{"ram_gb": "8", "color": None}]
    references = [{"ram_gb": "8", "color": None}]

    metrics = evaluate_attribute_extraction(predictions, references)

    assert metrics.overall.precision == 1.0
    assert metrics.overall.recall == 1.0
    assert metrics.overall.f1 == 1.0
    assert metrics.hallucination_rate == 0.0