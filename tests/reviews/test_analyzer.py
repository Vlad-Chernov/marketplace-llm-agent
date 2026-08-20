from datetime import date

import pytest

from marketplace_agent.domain.models import Review
from marketplace_agent.llm.base import FakeLLMClient, LLMResponse
from marketplace_agent.reviews.analyzer import (
    ReviewClassificationError,
    classify_review,
)
from marketplace_agent.reviews.taxonomy import (
    DefectCategory,
    DefectTaxonomy,
)


def test_classifies_product_defect_from_fixed_taxonomy() -> None:
    review = Review(
        review_id="REV-000001",
        sku="LAP-0001",
        rating=1,
        text="Ноутбук сильно перегревается во время работы.",
        created_at=date(2026, 1, 1),
        helpful_count=3,
    )
    taxonomy = DefectTaxonomy(
        categories=[
            DefectCategory(
                id="overheating",
                label="Перегрев",
                description="Ноутбук сильно нагревается во время работы.",
                examples=["Ноутбук сильно перегревается."],
            ),
            DefectCategory(
                id="other",
                label="Другой дефект",
                description="Неизвестный дефект товара.",
            ),
            DefectCategory(
                id="no_defect",
                label="Нет дефекта",
                description="Дефекта товара нет.",
            ),
        ]
    )
    client = FakeLLMClient(
        chat_responses=[
            LLMResponse(
                content=(
                    '{"defect_id":"overheating",'
                    '"kind":"product_defect",'
                    '"severity":"high",'
                    '"confidence":0.95}'
                ),
                model="fake-model",
                prompt_tokens=10,
                completion_tokens=5,
            )
        ]
    )

    labels = classify_review(review, taxonomy, client)

    assert labels.defect_id == "overheating"
    assert labels.kind == "product_defect"
    assert labels.severity == "high"
    assert labels.confidence == 0.95

def test_classifies_delivery_review_without_product_defect() -> None:
    review = Review(
        review_id="REV-000002",
        sku="LAP-0001",
        rating=3,
        text="Курьер задержал доставку.",
        created_at=date(2026, 1, 1),
        helpful_count=0,
    )
    taxonomy = DefectTaxonomy(
        categories=[
            DefectCategory(
                id="no_defect",
                label="Нет дефекта",
                description="Дефекта товара нет.",
            )
        ]
    )
    client = FakeLLMClient(
        chat_responses=[
            LLMResponse(
                content=(
                    '{"defect_id":"no_defect",'
                    '"kind":"delivery_or_service",'
                    '"severity":null,'
                    '"confidence":0.9}'
                ),
                model="fake-model",
                prompt_tokens=10,
                completion_tokens=5,
            )
        ]
    )

    labels = classify_review(review, taxonomy, client)

    assert labels.kind == "delivery_or_service"
    assert labels.defect_id == "no_defect"
    assert labels.severity is None


def test_rejects_unknown_defect_id() -> None:
    review = Review(
        review_id="REV-000003",
        sku="LAP-0001",
        rating=1,
        text="Неизвестная проблема.",
        created_at=date(2026, 1, 1),
        helpful_count=0,
    )
    taxonomy = DefectTaxonomy(
        categories=[
            DefectCategory(
                id="no_defect",
                label="Нет дефекта",
                description="Дефекта товара нет.",
            )
        ]
    )
    client = FakeLLMClient(
        chat_responses=[
            LLMResponse(
                content=(
                    '{"defect_id":"unknown_issue",'
                    '"kind":"product_defect",'
                    '"severity":"low",'
                    '"confidence":0.7}'
                ),
                model="fake-model",
                prompt_tokens=10,
                completion_tokens=5,
            )
        ]
    )

    with pytest.raises(
        ReviewClassificationError,
        match="Unknown defect ID",
    ):
        classify_review(review, taxonomy, client)