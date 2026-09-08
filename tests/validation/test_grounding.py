from typing import Any

import pytest

from marketplace_agent.domain.models import GeneratedContent
from marketplace_agent.llm.base import LLMResponse
from marketplace_agent.llm.structured import StructuredOutputError
from marketplace_agent.validation.grounding import (
    GroundingEvidence,
    validate_grounding,
)


class RecordingLLMClient:
    def __init__(self, content: str = '{"unsupported_claims": []}') -> None:
        self.content = content
        self.call_count = 0

    def chat(self, **_: Any) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(
            content=self.content,
            model="fake-model",
            prompt_tokens=10,
            completion_tokens=5,
        )


def make_content(used_attributes: dict[str, Any]) -> GeneratedContent:
    return GeneratedContent(
        title="Lenovo IdeaPad 14",
        bullets=["Экран 14 дюймов"],
        description="Ноутбук для учёбы и работы.",
        keywords=["ноутбук"],
        used_attributes=used_attributes,
    )


def make_evidence() -> GroundingEvidence:
    return GroundingEvidence(
        confirmed_attributes={"screen_size": 14, "ssd_gb": 512},
        supplier_description="Lenovo IdeaPad: экран 14 дюймов, SSD 512 ГБ.",
    )


def test_rejects_different_used_attribute_value_without_llm_call() -> None:
    client = RecordingLLMClient()

    result = validate_grounding(
        make_content({"screen_size": 14, "ssd_gb": 1024}),
        make_evidence(),
        client,
    )

    assert client.call_count == 0
    assert [claim.model_dump() for claim in result.unsupported_claims] == [
        {
            "field": "used_attributes",
            "claim": "ssd_gb=1024",
            "reason": "Значение не совпадает с подтверждённым: 512.",
        }
    ]


def test_accepts_numeric_string_matching_confirmed_number() -> None:
    client = RecordingLLMClient()

    result = validate_grounding(
        make_content({"screen_size": "14.0", "ssd_gb": "512"}),
        make_evidence(),
        client,
    )

    assert client.call_count == 1
    assert result.unsupported_claims == []


def test_rejects_unknown_used_attribute_without_llm_call() -> None:
    client = RecordingLLMClient()

    result = validate_grounding(
        make_content({"screen_size": 14, "battery_hours": 18}),
        make_evidence(),
        client,
    )

    assert client.call_count == 0
    assert [claim.model_dump() for claim in result.unsupported_claims] == [
        {
            "field": "used_attributes",
            "claim": "battery_hours=18",
            "reason": "Атрибут отсутствует в подтверждённых данных.",
        }
    ]

def test_detects_invented_text_claim_in_one_llm_call() -> None:
    client = RecordingLLMClient(
        content="""
        {
          "unsupported_claims": [
            {
              "field": "description",
              "claim": "работает до 18 часов",
              "reason": "В описании поставщика нет времени автономной работы."
            }
          ]
        }
        """
    )

    result = validate_grounding(
        GeneratedContent(
            title="Lenovo IdeaPad 14",
            bullets=["Экран 14 дюймов"],
            description="Работает до 18 часов без подзарядки.",
            keywords=["ноутбук"],
            used_attributes={"screen_size": 14, "ssd_gb": 512},
        ),
        make_evidence(),
        client,
    )

    assert client.call_count == 1
    assert [claim.model_dump() for claim in result.unsupported_claims] == [
        {
            "field": "description",
            "claim": "работает до 18 часов",
            "reason": "В описании поставщика нет времени автономной работы.",
        }
    ]


def test_accepts_card_when_llm_finds_no_unsupported_claims() -> None:
    client = RecordingLLMClient()

    result = validate_grounding(
        make_content({"screen_size": 14, "ssd_gb": 512}),
        make_evidence(),
        client,
    )

    assert client.call_count == 1
    assert result.unsupported_claims == []


def test_rejects_llm_claim_with_field_outside_schema() -> None:
    client = RecordingLLMClient(
        content="""
        {
          "unsupported_claims": [
            {
              "field": "price",
              "claim": "Цена снижена",
              "reason": "Поле не разрешено схемой."
            }
          ]
        }
        """
    )

    with pytest.raises(StructuredOutputError):
        validate_grounding(
            make_content({"screen_size": 14, "ssd_gb": 512}),
            make_evidence(),
            client,
        )
