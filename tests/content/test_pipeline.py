import json
from decimal import Decimal

import pytest

from marketplace_agent.content.pipeline import run_content_pipeline
from marketplace_agent.domain.models import Product
from marketplace_agent.llm.base import FakeLLMClient, LLMResponse


def response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="fake-model",
        prompt_tokens=10,
        completion_tokens=10,
    )


def make_product() -> Product:
    return Product(
        sku="LAP-0001",
        category="laptops",
        brand="Lenovo",
        model="IdeaPad 1000",
        price=Decimal(75000),
        sales_count=10,
        supplier_description=(
            "Ноутбук Lenovo IdeaPad: экран 14 дюймов, SSD 512 ГБ."
        ),
        attributes={},
    )


def test_completes_pipeline_on_first_valid_attempt() -> None:
    client = FakeLLMClient(
        chat_responses=[
            response(
                """
                {
                  "attributes": {
                    "screen_diagonal_in": {
                      "value": 14.0,
                      "confidence": 0.95
                    },
                    "storage_gb": {
                      "value": "512",
                      "confidence": 0.95
                    }
                  },
                  "unsupported_facts": []
                }
                """
            ),
            response(
                """
                {
                  "title": "Ноутбук Lenovo для работы и учёбы",
                  "bullets": ["Экран 14 дюймов", "SSD 512 ГБ"],
                  "description": "Ноутбук Lenovo с экраном 14 дюймов и SSD 512 ГБ.",
                  "keywords": ["ноутбук", "Lenovo"],
                  "used_attributes": {
                    "screen_diagonal_in": 14.0,
                    "storage_gb": "512"
                  }
                }
                """
            ),
            response('{"unsupported_claims": []}'),
            response('{"violations": []}'),
        ]
    )

    result = run_content_pipeline(make_product(), client)

    assert result.status == "completed"
    assert result.attempts == 1
    assert result.violations == []
    assert result.content is not None
    assert result.content.title == "Ноутбук Lenovo для работы и учёбы"

def test_repairs_invalid_content_on_second_attempt() -> None:
    client = FakeLLMClient(
        chat_responses=[
            response(
                """
                {
                  "attributes": {
                    "screen_diagonal_in": {
                      "value": 14.0,
                      "confidence": 0.95
                    },
                    "storage_gb": {
                      "value": "512",
                      "confidence": 0.95
                    }
                  },
                  "unsupported_facts": []
                }
                """
            ),
            response(
                """
                {
                  "title": "Коротко",
                  "bullets": ["Экран 14 дюймов"],
                  "description": "Ноутбук Lenovo с экраном 14 дюймов и SSD 512 ГБ.",
                  "keywords": ["ноутбук"],
                  "used_attributes": {
                    "screen_diagonal_in": 14.0,
                    "storage_gb": "512"
                  }
                }
                """
            ),
            response('{"unsupported_claims": []}'),
            response('{"violations": []}'),
            response(
                """
                {
                  "title": "Ноутбук Lenovo для работы и учёбы",
                  "bullets": ["Экран 14 дюймов", "SSD 512 ГБ"],
                  "description": "Ноутбук Lenovo с экраном 14 дюймов и SSD 512 ГБ.",
                  "keywords": ["ноутбук", "Lenovo"],
                  "used_attributes": {
                    "screen_diagonal_in": 14.0,
                    "storage_gb": "512"
                  }
                }
                """
            ),
            response('{"unsupported_claims": []}'),
            response('{"violations": []}'),
        ]
    )

    result = run_content_pipeline(make_product(), client)

    assert result.status == "completed"
    assert result.attempts == 2
    assert result.violations == []
    assert result.content is not None
    assert result.content.title == "Ноутбук Lenovo для работы и учёбы"

def test_sends_repeated_invalid_content_to_manual_review() -> None:
    extracted_attributes = """
    {
      "attributes": {
        "screen_diagonal_in": {"value": 14.0, "confidence": 0.95},
        "storage_gb": {"value": "512", "confidence": 0.95}
      },
      "unsupported_facts": []
    }
    """
    invalid_content = """
    {
      "title": "Коротко",
      "bullets": ["Экран 14 дюймов"],
      "description": "Ноутбук Lenovo с экраном 14 дюймов и SSD 512 ГБ.",
      "keywords": ["ноутбук"],
      "used_attributes": {
        "screen_diagonal_in": 14.0,
        "storage_gb": "512"
      }
    }
    """
    client = FakeLLMClient(
        chat_responses=[
            response(extracted_attributes),
            response(invalid_content),
            response('{"unsupported_claims": []}'),
            response('{"violations": []}'),
            response(invalid_content),
            response('{"unsupported_claims": []}'),
            response('{"violations": []}'),
        ]
    )

    result = run_content_pipeline(make_product(), client)

    assert result.status == "manual_review"
    assert result.attempts == 2
    assert [violation.rule_id for violation in result.violations] == [
        "title-length"
    ]


def test_rejects_non_positive_attempt_limit() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        run_content_pipeline(
            make_product(),
            FakeLLMClient(),
            max_attempts=0,
        )

def test_sends_three_distinct_invalid_cards_to_manual_review() -> None:
    extracted_attributes = {
        "attributes": {
            "screen_diagonal_in": {
                "value": 14.0,
                "confidence": 0.95,
            },
            "storage_gb": {
                "value": "512",
                "confidence": 0.95,
            },
        },
        "unsupported_facts": [],
    }

    def invalid_content(title: str) -> str:
        return json.dumps(
            {
                "title": title,
                "bullets": ["Экран 14 дюймов"],
                "description": (
                    "Ноутбук Lenovo с экраном 14 дюймов и SSD 512 ГБ."
                ),
                "keywords": ["ноутбук"],
                "used_attributes": {
                    "screen_diagonal_in": 14.0,
                    "storage_gb": "512",
                },
            },
            ensure_ascii=False,
        )

    client = FakeLLMClient(
        chat_responses=[
            response(json.dumps(extracted_attributes)),
            response(invalid_content("Коротко 1")),
            response('{"unsupported_claims": []}'),
            response('{"violations": []}'),
            response(invalid_content("Коротко 2")),
            response('{"unsupported_claims": []}'),
            response('{"violations": []}'),
            response(invalid_content("Коротко 3")),
            response('{"unsupported_claims": []}'),
            response('{"violations": []}'),
        ]
    )

    result = run_content_pipeline(make_product(), client)

    assert result.status == "manual_review"
    assert result.attempts == 3
    assert [violation.rule_id for violation in result.violations] == [
        "title-length"
    ]