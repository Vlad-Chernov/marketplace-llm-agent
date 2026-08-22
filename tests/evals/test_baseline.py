from marketplace_agent.evals.baseline import build_messages
from marketplace_agent.evals.models import GoldenCase


def test_builds_strict_prompt_for_attribute_extraction() -> None:
    case = GoldenCase(
        id="gold-001",
        question="Из описания «Диагональ 14.0, ОЗУ 16 ГБ» извлеки атрибуты.",
        type="attribute_extraction",
        input={"supplier_description": "Ноутбук с ОЗУ 16 ГБ."},
        expected_answer={
            "screen_diagonal_in": 14.0,
            "ram_gb": "16",
        },
        origin="manual",
    )

    messages = build_messages(case)

    assert messages[1].content == case.question
    assert "screen_diagonal_in" in messages[0].content
    assert "storage_gb" in messages[0].content
    assert "diagonal_inches" not in messages[0].content
    assert "только JSON" in messages[0].content