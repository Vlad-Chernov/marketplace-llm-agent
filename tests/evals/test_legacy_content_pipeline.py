from decimal import Decimal
from pathlib import Path

from marketplace_agent.domain.models import Product
from marketplace_agent.evals.legacy_content_pipeline import (
    run_legacy_content_pipeline,
)
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


def test_legacy_pipeline_completes_first_valid_attempt() -> None:
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

    result = run_legacy_content_pipeline(make_product(), client)

    assert result.status == "completed"
    assert result.attempts == 1


def test_production_pipeline_does_not_import_legacy() -> None:
    source = Path(
        "src/marketplace_agent/content/pipeline.py"
    ).read_text(encoding="utf-8")

    assert "legacy_content_pipeline" not in source
