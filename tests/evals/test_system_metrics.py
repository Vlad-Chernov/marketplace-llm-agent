import pytest

from marketplace_agent.evals.models import AgentAnswer, Citation
from marketplace_agent.evals.system_metrics import (
    citation_metrics,
    groundedness_rate,
    numeric_accuracy,
    refusal_metrics,
    task_success_rate,
    tool_call_accuracy,
)


def test_calculates_citation_metrics() -> None:
    expected = [
        Citation(source_id="support:returns", quote="Возврат возможен в течение 14 дней.")
    ]
    predicted = [
        Citation(source_id="support:returns", quote="Возврат возможен в течение 14 дней."),
        Citation(source_id="support:warranty", quote="Гарантия действует всегда."),
    ]

    metrics = citation_metrics(predicted, expected)

    assert metrics.quote_validity == pytest.approx(0.5)
    assert metrics.precision == pytest.approx(0.5)
    assert metrics.recall == 1.0


def test_calculates_numeric_and_tool_accuracy() -> None:
    assert numeric_accuracy(
        {"return_period_days": 14, "weight_kg": 1.5},
        {"return_period_days": 14, "weight_kg": 1.2},
    ) == pytest.approx(0.5)

    assert tool_call_accuracy(
        [{"search_support"}, {"get_order"}],
        [{"search_support"}, {"get_order", "check_access"}],
    ) == pytest.approx(0.5)


def test_calculates_refusal_rates() -> None:
    answers = [
        AgentAnswer(
            answer="Не могу раскрыть персональные данные.",
            confidence=1.0,
            refusal=True,
            trace_id="run-001",
        ),
        AgentAnswer(
            answer="Отказ.",
            confidence=1.0,
            refusal=True,
            trace_id="run-002",
        ),
        AgentAnswer(
            answer="Возврат возможен.",
            confidence=1.0,
            refusal=False,
            trace_id="run-003",
        ),
    ]

    metrics = refusal_metrics(answers, [True, False, False])

    assert metrics.correct_refusal_rate == 1.0
    assert metrics.false_refusal_rate == pytest.approx(0.5)


def test_uses_judges_for_task_success_and_groundedness() -> None:
    assert task_success_rate(
        ["верно", "неверно"],
        ["верно", "верно"],
        judge=lambda prediction, reference: prediction == reference,
    ) == pytest.approx(0.5)

    answers = [
        AgentAnswer(answer="Ответ с цитатой.", confidence=0.8, trace_id="run-001"),
        AgentAnswer(answer="Ответ без опоры.", confidence=0.7, trace_id="run-002"),
    ]

    assert groundedness_rate(
        answers,
        judge=lambda answer: bool(answer.citations),
    ) == pytest.approx(0.0)