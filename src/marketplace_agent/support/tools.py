from collections.abc import Mapping
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from marketplace_agent.retrieval.lexical import SearchResult
from marketplace_agent.storage.repositories import (
    OrderRepository,
    ProductRepository,
)


class ToolResult(BaseModel):
    """Store a safe result from a read-only support tool."""

    model_config = ConfigDict(extra="forbid")

    ok: bool
    data: dict[str, Any] | list[dict[str, Any]] | None = None
    error: str | None = None
    error_code: str | None = None
    citations: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


class Tool(Protocol):
    """Describe one registered support tool."""

    name: str
    description: str
    input_model: type[BaseModel]

    def run(self, **arguments: Any) -> ToolResult: ...


class GetProductInput(BaseModel):
    """Validate product lookup arguments."""

    model_config = ConfigDict(extra="forbid")

    sku: str = Field(min_length=1)


class GetProductTool:
    """Read public product information by SKU."""

    name = "get_product"
    description = "Получить публичные данные товара по SKU."
    input_model = GetProductInput

    def __init__(self, repository: ProductRepository) -> None:
        self._repository = repository

    def run(self, **arguments: Any) -> ToolResult:
        try:
            validated = GetProductInput.model_validate(arguments)
        except ValueError:
            return ToolResult(
                ok=False,
                error="Некорректный SKU.",
                error_code="invalid_arguments",
            )

        product = self._repository.get_by_sku(validated.sku)
        if product is None:
            return ToolResult(
                ok=False,
                error="Товар не найден.",
                error_code="not_found",
            )

        return ToolResult(
            ok=True,
            data={
                "sku": product.sku,
                "category": product.category,
                "brand": product.brand,
                "model": product.model,
                "price": str(product.price),
                "attributes": product.attributes,
            },
        )

class GetOrderInput(BaseModel):
    """Validate session-scoped order lookup arguments."""

    model_config = ConfigDict(extra="forbid")

    order_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)


class GetOrderTool:
    """Read an order only for its owner session."""

    name = "get_order"
    description = "Получить заказ только для текущей сессии."
    input_model = GetOrderInput

    def __init__(self, repository: OrderRepository) -> None:
        self._repository = repository

    def run(self, **arguments: Any) -> ToolResult:
        try:
            validated = GetOrderInput.model_validate(arguments)
        except ValueError:
            return ToolResult(
                ok=False,
                error="Некорректные параметры заказа.",
                error_code="invalid_arguments",
            )

        order = self._repository.get_by_id(validated.order_id)
        if order is None:
            return ToolResult(
                ok=False,
                error="Заказ не найден.",
                error_code="not_found",
            )
        if order.session_id != validated.session_id:
            return ToolResult(
                ok=False,
                error="Доступ к заказу запрещён.",
                error_code="access_denied",
            )

        return ToolResult(
            ok=True,
            data={
                "order_id": order.order_id,
                "sku": order.sku,
                "status": order.status,
                "price": str(order.price),
                "purchased_at": order.purchased_at.isoformat(),
                "delivered_at": (
                    order.delivered_at.isoformat()
                    if order.delivered_at is not None
                    else None
                ),
            },
        )

class PolicyRetriever(Protocol):
    """Search policy chunks."""

    def search(
        self,
        query: str,
        k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[SearchResult]: ...


class SearchPolicyInput(BaseModel):
    """Validate policy-search arguments."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)


class SearchPolicyTool:
    """Search support policies and return citable chunks."""

    name = "search_policy"
    description = "Найти правила поддержки по вопросу покупателя."
    input_model = SearchPolicyInput

    def __init__(self, retriever: PolicyRetriever) -> None:
        self._retriever = retriever

    def run(self, **arguments: Any) -> ToolResult:
        try:
            validated = SearchPolicyInput.model_validate(arguments)
        except ValueError:
            return ToolResult(
                ok=False,
                error="Некорректный поисковый запрос.",
                error_code="invalid_arguments",
            )

        results = self._retriever.search(validated.query, k=3)
        return ToolResult(
            ok=True,
            data=[
                {
                    "chunk_id": result.chunk.chunk_id,
                    "heading": result.chunk.heading,
                    "text": result.chunk.text,
                }
                for result in results
            ],
            citations=[result.chunk.chunk_id for result in results],
        )