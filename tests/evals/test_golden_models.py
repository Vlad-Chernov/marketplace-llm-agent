import pytest
from pydantic import ValidationError

from marketplace_agent.evals.models import GoldenCase


def test_creates_golden_case() -> None:
    case = GoldenCase(
        id="gold-001",
        question="Извлеки объём оперативной памяти.",
        type="attribute_extraction",
        expected_answer={"ram_gb": "16"},
        expected_citations=[],
        must_call_tools=[],
        must_not_contain=["32"],
        origin="manual",
        notes="Проверка точного извлечения.",
    )

    assert case.id == "gold-001"
    assert case.expected_answer == {"ram_gb": "16"}


@pytest.mark.parametrize(
    "data",
    [
        {
            "id": "gold-001",
            "question": "Вопрос",
            "expected_answer": {"ram_gb": "16"},
            "origin": "manual",
        },
        {
            "id": "gold-001",
            "question": "Вопрос",
            "type": "attribute_extraction",
            "origin": "manual",
        },
    ],
)
def test_rejects_case_without_type_or_expected_answer(data: dict) -> None:
    with pytest.raises(ValidationError):
        GoldenCase(**data)