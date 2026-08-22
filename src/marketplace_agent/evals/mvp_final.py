import json
from collections.abc import Callable, Sequence
from pathlib import Path

from marketplace_agent.evals.models import GoldenCase
from marketplace_agent.evals.runner import (
    EvaluationPrediction,
    EvaluationRun,
    run_evaluation,
)


def load_golden_cases(path: Path) -> list[GoldenCase]:
    """Загрузить и проверить JSON-набор эталонных кейсов."""

    raw_cases = json.loads(path.read_text(encoding="utf-8"))
    return [
        GoldenCase.model_validate(raw_case)
        for raw_case in raw_cases
    ]


def run_mvp_final(
    cases: Sequence[GoldenCase],
    execute_case: Callable[[GoldenCase], EvaluationPrediction],
) -> EvaluationRun:
    """Запустить полный MVP-набор через переданный executor."""

    return run_evaluation(
        cases=cases,
        suite="mvp-final",
        version="MVP-final",
        execute_case=execute_case,
    )