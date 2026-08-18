from marketplace_agent.content.repair import repair_content
from marketplace_agent.domain.models import GeneratedContent, RuleViolation
from marketplace_agent.llm.base import FakeLLMClient, LLMResponse
from marketplace_agent.validation.grounding import GroundingEvidence


def test_repairs_content_from_explicit_violations() -> None:
    client = FakeLLMClient(
        chat_responses=[
            LLMResponse(
                content=(
                    '{"title":"Ноутбук Lenovo для работы и учёбы",'
                    '"bullets":["Экран 14 дюймов"],'
                    '"description":"Ноутбук с SSD 512 ГБ.",'
                    '"keywords":["ноутбук","Lenovo"],'
                    '"used_attributes":{"screen_diagonal_in":14.0,'
                    '"storage_gb":"512"}}'
                ),
                model="fake-model",
                prompt_tokens=10,
                completion_tokens=10,
            )
        ]
    )
    content = GeneratedContent(
        title="Коротко",
        bullets=["Экран 14 дюймов"],
        description="Ноутбук с SSD 512 ГБ.",
        keywords=["ноутбук"],
        used_attributes={
            "screen_diagonal_in": 14.0,
            "storage_gb": "512",
        },
    )
    violations = [
        RuleViolation(
            rule_id="title-length",
            message="Заголовок должен содержать от 20 до 120 символов.",
            severity="medium",
            field="title",
        )
    ]
    evidence = GroundingEvidence(
        confirmed_attributes={
            "screen_diagonal_in": 14.0,
            "storage_gb": "512",
        },
        supplier_description="Ноутбук Lenovo: экран 14 дюймов, SSD 512 ГБ.",
    )

    result = repair_content(content, violations, evidence, client)

    assert result.title == "Ноутбук Lenovo для работы и учёбы"
    assert result.used_attributes == {
        "screen_diagonal_in": 14.0,
        "storage_gb": "512",
    }