from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar

from marketplace_agent.evals.models import AgentAnswer, Citation

TaskValue = TypeVar("TaskValue")


@dataclass(frozen=True)
class CitationMetrics:
    quote_validity: float
    precision: float
    recall: float


@dataclass(frozen=True)
class RefusalMetrics:
    correct_refusal_rate: float
    false_refusal_rate: float


def citation_metrics(
    predicted: Sequence[Citation],
    expected: Sequence[Citation],
) -> CitationMetrics:
    valid_quotes = sum(citation in expected for citation in predicted)
    quote_validity = valid_quotes / len(predicted) if predicted else 0.0

    predicted_sources = {citation.source_id for citation in predicted}
    expected_sources = {citation.source_id for citation in expected}
    matching_sources = predicted_sources & expected_sources

    precision = (
        len(matching_sources) / len(predicted_sources)
        if predicted_sources
        else 0.0
    )
    recall = (
        len(matching_sources) / len(expected_sources)
        if expected_sources
        else 0.0
    )

    return CitationMetrics(
        quote_validity=quote_validity,
        precision=precision,
        recall=recall,
    )


def numeric_accuracy(
    predicted: Mapping[str, float],
    expected: Mapping[str, float],
) -> float:
    if not expected:
        return 0.0

    correct_count = sum(
        predicted.get(name) == value for name, value in expected.items()
    )
    return correct_count / len(expected)


def tool_call_accuracy(
    called_tools: Sequence[set[str]],
    expected_tools: Sequence[set[str]],
) -> float:
    if len(called_tools) != len(expected_tools):
        raise ValueError("Tool-call sequences must have the same length.")
    if not expected_tools:
        return 0.0

    correct_count = sum(
        called == expected
        for called, expected in zip(called_tools, expected_tools, strict=True)
    )
    return correct_count / len(expected_tools)


def refusal_metrics(
    answers: Sequence[AgentAnswer],
    expected_refusals: Sequence[bool],
) -> RefusalMetrics:
    if len(answers) != len(expected_refusals):
        raise ValueError("Answers and expected refusals must have the same length.")

    expected_refusal_count = sum(expected_refusals)
    expected_answer_count = len(expected_refusals) - expected_refusal_count

    correct_refusal_count = sum(
        answer.refusal and expected_refusal
        for answer, expected_refusal in zip(
            answers,
            expected_refusals,
            strict=True,
        )
    )
    false_refusal_count = sum(
        answer.refusal and not expected_refusal
        for answer, expected_refusal in zip(
            answers,
            expected_refusals,
            strict=True,
        )
    )

    return RefusalMetrics(
        correct_refusal_rate=(
            correct_refusal_count / expected_refusal_count
            if expected_refusal_count
            else 0.0
        ),
        false_refusal_rate=(
            false_refusal_count / expected_answer_count
            if expected_answer_count
            else 0.0
        ),
    )


def task_success_rate(
    predictions: Sequence[TaskValue],
    references: Sequence[TaskValue],
    judge: Callable[[TaskValue, TaskValue], bool],
) -> float:
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")
    if not references:
        return 0.0

    success_count = sum(
        judge(prediction, reference)
        for prediction, reference in zip(predictions, references, strict=True)
    )
    return success_count / len(references)


def groundedness_rate(
    answers: Sequence[AgentAnswer],
    judge: Callable[[AgentAnswer], bool],
) -> float:
    if not answers:
        return 0.0

    grounded_count = sum(judge(answer) for answer in answers)
    return grounded_count / len(answers)