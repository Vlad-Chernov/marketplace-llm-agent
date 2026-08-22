import pytest
from pydantic import ValidationError

from marketplace_agent.evals.models import AgentAnswer, Citation, GoldenCase


def test_creates_golden_case() -> None:
    case = GoldenCase(
        id="gold-001",
        question="Извлеки объём оперативной памяти.",
        type="attribute_extraction",
        input={"supplier_description": "Ноутбук с ОЗУ 16 ГБ."},
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
        {
            "id": "gold-001",
            "question": "Вопрос",
            "type": "attribute_extraction",
            "expected_answer": {"ram_gb": "16"},
            "origin": "manual",
        },
    ],
)



def test_rejects_case_without_type_or_expected_answer(data: dict) -> None:
    with pytest.raises(ValidationError):
        GoldenCase(**data)

def test_stores_structured_input() -> None:
    case = GoldenCase(
        id="gold-002",
        question="Извлеки характеристики товара.",
        type="attribute_extraction",
        input={
            "supplier_description": "Ноутбук: ОЗУ 16 ГБ, SSD 512 ГБ.",
        },
        expected_answer={"ram_gb": "16", "storage_gb": "512"},
        origin="manual",
    )

    assert case.input == {
        "supplier_description": "Ноутбук: ОЗУ 16 ГБ, SSD 512 ГБ.",
    }

def test_creates_agent_answer_with_traceability() -> None:
    answer = AgentAnswer(
        answer="Возврат возможен в течение 14 дней после получения.",
        citations=[
            Citation(
                source_id="support:returns",
                quote="Покупатель может вернуть ноутбук надлежащего качества в течение 14 календарных дней после получения.",
            )
        ],
        numbers={"return_period_days": 14},
        confidence=0.9,
        refusal=False,
        trace_id="run-001",
    )

    assert answer.citations[0].source_id == "support:returns"
    assert answer.numbers["return_period_days"] == 14