from typing import Any

import pytest

from marketplace_agent.domain.models import GeneratedContent
from marketplace_agent.llm.base import FakeLLMClient, LLMResponse
from marketplace_agent.validation.semantic import (
    SemanticValidationError,
    validate_semantic,
)

SEMANTIC_RULES = [
    {
        "id": "medical-claim",
        "description": "Нельзя заявлять о лечебном или медицинском эффекте товара.",
        "severity": "critical",
    },
    {
        "id": "unverifiable-superlative",
        "description": "Нельзя использовать недоказуемые превосходные степени.",
        "severity": "medium",
    },
    {
        "id": "competitor-comparison",
        "description": "Нельзя сравнивать товар с конкурентами.",
        "severity": "medium",
    },
]


class RecordingLLMClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.call_count = 0
        self.messages: list[Any] = []

    def chat(self, **_: Any) -> LLMResponse:
        self.messages = _["messages"]
        self.call_count += 1
        return LLMResponse(
            content=self.content,
            model="fake-model",
            prompt_tokens=10,
            completion_tokens=5,
        )


def make_content(title: str, description: str) -> GeneratedContent:
    return GeneratedContent(
        title=title,
        bullets=["Экран 14 дюймов"],
        description=description,
        keywords=["ноутбук"],
        used_attributes={},
    )


def test_detects_semantic_violations_in_one_llm_call() -> None:
    client = RecordingLLMClient(
        content="""
        {
          "violations": [
            {
              "rule_id": "medical-claim",
              "field": "description",
              "evidence": "лечит усталость глаз"
            },
            {
              "rule_id": "unverifiable-superlative",
              "field": "title",
              "evidence": "Лучший ноутбук"
            },
            {
              "rule_id": "competitor-comparison",
              "field": "description",
              "evidence": "лучше моделей конкурентов"
            }
          ]
        }
        """
    )
    content = make_content(
        title="Лучший ноутбук Lenovo",
        description="Он лечит усталость глаз и лучше моделей конкурентов.",
    )

    violations = validate_semantic(content, SEMANTIC_RULES, client)

    assert client.call_count == 1
    assert {violation.rule_id for violation in violations} == {
        "medical-claim",
        "unverifiable-superlative",
        "competitor-comparison",
    }
    assert {violation.severity for violation in violations} == {
        "critical",
        "medium",
    }


def test_accepts_safe_content() -> None:
    client = RecordingLLMClient(content='{"violations": []}')
    content = make_content(
        title="Lenovo IdeaPad 14",
        description="Ноутбук с экраном 14 дюймов и SSD 512 ГБ.",
    )

    assert validate_semantic(content, SEMANTIC_RULES, client) == []
    assert client.call_count == 1


def test_rejects_violation_with_unknown_rule_id() -> None:
    client = RecordingLLMClient(
        content="""
        {
          "violations": [
            {
              "rule_id": "unknown-rule",
              "field": "title",
              "evidence": "Текст"
            }
          ]
        }
        """
    )
    content = make_content(
        title="Lenovo IdeaPad 14",
        description="Описание.",
    )

    with pytest.raises(SemanticValidationError, match="unknown rule"):
        validate_semantic(content, SEMANTIC_RULES, client)

def test_retries_once_after_invalid_llm_response() -> None:
    content = GeneratedContent(
        title="Ноутбук для работы и учебы",
        bullets=[],
        description="Лучший ноутбук для работы.",
        keywords=[],
        used_attributes={},
    )
    rules = [
        {
            "id": "unverifiable-superlative",
            "description": "Нельзя использовать недоказуемые превосходные степени.",
            "severity": "medium",
        }
    ]
    client = FakeLLMClient(
        [
            LLMResponse(
                content="Это не JSON",
                model="fake-model",
                prompt_tokens=10,
                completion_tokens=5,
            ),
            LLMResponse(
                content=(
                    '{"violations":['
                    '{"rule_id":"unverifiable-superlative",'
                    '"field":"description",'
                    '"evidence":"Лучший"}'
                    ']}'
                ),
                model="fake-model",
                prompt_tokens=10,
                completion_tokens=5,
            ),
        ]
    )

    violations = validate_semantic(content, rules, client)

    assert [violation.rule_id for violation in violations] == [
        "unverifiable-superlative"
    ]

def test_prompt_classifies_eye_health_claim_as_medical() -> None:
    client = RecordingLLMClient(content='{"violations": []}')
    content = make_content(
        title="Ноутбук для работы",
        description="Ноутбук для здоровья глаз.",
    )

    validate_semantic(content, SEMANTIC_RULES, client)

    prompt = client.messages[0].content
    assert "здоровье глаз" in prompt
    assert "medical-claim" in prompt