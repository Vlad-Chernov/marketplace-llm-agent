from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from marketplace_agent.domain.models import Order, Product
from marketplace_agent.retrieval.documents import PolicyChunk
from marketplace_agent.retrieval.lexical import SearchResult
from marketplace_agent.storage.database import initialize_database
from marketplace_agent.storage.repositories import OrderRepository, ProductRepository
from marketplace_agent.support.tools import (
    GetOrderTool,
    GetProductTool,
    SearchPolicyTool,
)


def make_product() -> Product:
    return Product(
        sku="LAP-0001",
        category="laptop",
        brand="Lenovo",
        model="IdeaPad",
        price=Decimal(79990),
        sales_count=10,
        supplier_description="Описание",
        attributes={"ram_gb": "16"},
        true_attributes={"ram_gb": "16"},
    )


def test_returns_public_product_data(tmp_path: Path) -> None:
    database_path = tmp_path / "marketplace.db"
    initialize_database(database_path)
    repository = ProductRepository(database_path)
    repository.save_many([make_product()])

    result = GetProductTool(repository).run(sku="LAP-0001")

    assert result.ok is True
    assert result.data == {
        "sku": "LAP-0001",
        "category": "laptop",
        "brand": "Lenovo",
        "model": "IdeaPad",
        "price": "79990",
        "attributes": {"ram_gb": "16"},
    }
    assert result.error is None


def test_returns_safe_errors_for_unknown_or_blank_sku(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "marketplace.db"
    initialize_database(database_path)
    tool = GetProductTool(ProductRepository(database_path))

    assert tool.run(sku="UNKNOWN").error_code == "not_found"
    assert tool.run(sku="").error_code == "invalid_arguments"

def make_order() -> Order:
    return Order(
        order_id="ORD-000001",
        session_id="session-owner",
        sku="LAP-0001",
        status="delivered",
        price=Decimal(79990),
        purchased_at=datetime(2026, 8, 1, tzinfo=UTC),
        delivered_at=datetime(2026, 8, 3, tzinfo=UTC),
    )


def test_returns_order_only_to_its_session_owner(tmp_path: Path) -> None:
    database_path = tmp_path / "marketplace.db"
    initialize_database(database_path)
    repository = OrderRepository(database_path)
    repository.save_many([make_order()])
    tool = GetOrderTool(repository)

    owner_result = tool.run(
        order_id="ORD-000001",
        session_id="session-owner",
    )
    foreign_result = tool.run(
        order_id="ORD-000001",
        session_id="session-foreign",
    )

    assert owner_result.ok is True
    assert owner_result.data["order_id"] == "ORD-000001"
    assert foreign_result.model_dump() == {
        "ok": False,
        "data": None,
        "error": "Доступ к заказу запрещён.",
        "error_code": "access_denied",
        "citations": [],
        "meta": {},
    }

class StubPolicyRetriever:
    def search(
        self,
        query: str,
        k: int,
        filters: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        if not query.strip():
            return []

        return [
            SearchResult(
                chunk=PolicyChunk(
                    chunk_id="returns-01",
                    document_id="returns",
                    policy_type="returns",
                    heading="Возврат",
                    text="Возврат возможен в течение 14 дней.",
                ),
                score=1.0,
                rank=1,
            )
        ]


def test_searches_policy_and_returns_citations() -> None:
    tool = SearchPolicyTool(StubPolicyRetriever())

    result = tool.run(query="Сколько дней можно вернуть товар?")

    assert result.ok is True
    assert result.citations == ["returns-01"]
    assert result.data == [
        {
            "chunk_id": "returns-01",
            "heading": "Возврат",
            "text": "Возврат возможен в течение 14 дней.",
        }
    ]
    assert tool.run(query="").error_code == "invalid_arguments"