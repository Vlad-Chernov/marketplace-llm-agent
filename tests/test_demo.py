from datetime import date
from decimal import Decimal
from pathlib import Path

from marketplace_agent.domain.models import (
    GeneratedContent,
    PipelineResult,
    Product,
    Review,
)
from marketplace_agent.reviews.analyzer import ReviewLabels
from marketplace_agent.storage.database import initialize_database
from marketplace_agent.storage.repositories import (
    ProductRepository,
    ReviewRepository,
)
from marketplace_agent.support.agent import AgentAnswer


def make_product() -> Product:
    return Product(
        sku="LAP-0001",
        category="laptops",
        brand="Lenovo",
        model="IdeaPad",
        price=Decimal(79990),
        sales_count=10,
        supplier_description="Ноутбук Lenovo: экран 14 дюймов, SSD 512 ГБ.",
        attributes={},
    )


def make_review() -> Review:
    return Review(
        review_id="REV-000001",
        sku="LAP-0001",
        rating=1,
        text="Ноутбук сильно перегревается.",
        created_at=date(2026, 8, 20),
        helpful_count=3,
    )


def test_runs_card_review_and_support_scenarios(
    monkeypatch,
    tmp_path: Path,
) -> None:
    from marketplace_agent import demo

    database_path = tmp_path / "marketplace.db"
    initialize_database(database_path)
    ProductRepository(database_path).save_many([make_product()])
    ReviewRepository(database_path).save_many([make_review()])

    monkeypatch.setattr(
        demo,
        "run_content_pipeline",
        lambda product, llm: PipelineResult(
            sku=product.sku,
            content=GeneratedContent(
                title="Ноутбук Lenovo",
                bullets=["Экран 14 дюймов"],
                description="Ноутбук для работы.",
                keywords=["ноутбук"],
                used_attributes={},
            ),
            attempts=1,
            status="completed",
        ),
    )
    monkeypatch.setattr(
        demo,
        "classify_review",
        lambda review, taxonomy, llm: ReviewLabels(
            defect_id="overheating",
            kind="product_defect",
            severity="high",
            confidence=0.95,
        ),
    )

    class StubSupportAgent:
        def __init__(self, registry, llm) -> None:
            del registry, llm

        def run(self, message, session_id, history) -> AgentAnswer:
            assert message == "Сколько дней можно вернуть товар?"
            assert session_id == "demo-session"
            assert history == []
            return AgentAnswer(
                status="answered",
                text="Вернуть товар можно в течение 14 дней.",
                citations=["returns-01"],
            )

    monkeypatch.setattr(demo, "SupportAgent", StubSupportAgent)

    result = demo.run_demo(
        database_path=database_path,
        retriever=object(),
        llm=object(),
    )

    assert result.product_sku == "LAP-0001"
    assert result.content.status == "completed"
    assert result.review.defect_id == "overheating"
    assert result.support.citations == ["returns-01"]

def test_formats_demo_result() -> None:
    from marketplace_agent import demo

    result = demo.DemoResult(
        product_sku="LAP-0001",
        content=PipelineResult(
            sku="LAP-0001",
            content=GeneratedContent(
                title="Ноутбук Lenovo",
                bullets=["Экран 14 дюймов"],
                description="Ноутбук для работы.",
                keywords=["ноутбук"],
                used_attributes={},
            ),
            attempts=1,
            status="completed",
        ),
        review=ReviewLabels(
            defect_id="overheating",
            kind="product_defect",
            severity="high",
            confidence=0.95,
        ),
        support=AgentAnswer(
            status="answered",
            text="Вернуть товар можно в течение 14 дней.",
            citations=["returns-01"],
        ),
    )

    assert demo.format_demo_result(result) == (
        "=== Карточка ===\n"
        "SKU: LAP-0001\n"
        "Статус: completed\n"
        "Заголовок: Ноутбук Lenovo\n\n"
        "=== Отзыв ===\n"
        "Дефект: overheating\n"
        "Серьёзность: high\n\n"
        "=== Поддержка ===\n"
        "Статус: answered\n"
        "Ответ: Вернуть товар можно в течение 14 дней.\n"
        "Цитаты: returns-01"
    )