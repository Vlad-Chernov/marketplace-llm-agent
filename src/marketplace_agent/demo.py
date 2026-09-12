from pathlib import Path

from pydantic import BaseModel, ConfigDict

from marketplace_agent.content.pipeline import run_content_pipeline
from marketplace_agent.domain.models import PipelineResult
from marketplace_agent.llm.base import LLMClient
from marketplace_agent.llm.telemetry import trace_event
from marketplace_agent.reviews.analyzer import ReviewLabels, classify_review
from marketplace_agent.reviews.taxonomy import load_defect_taxonomy
from marketplace_agent.storage.repositories import (
    OrderRepository,
    ProductRepository,
    ReviewRepository,
)
from marketplace_agent.support.agent import AgentAnswer, SupportAgent
from marketplace_agent.support.registry import ToolRegistry
from marketplace_agent.support.tools import (
    GetOrderTool,
    GetProductTool,
    PolicyRetriever,
    SearchPolicyTool,
)


class DemoResult(BaseModel):
    """Store the results of the three MVP demonstration scenarios."""

    model_config = ConfigDict(extra="forbid")

    product_sku: str
    content: PipelineResult
    review: ReviewLabels
    support: AgentAnswer


def run_demo(
    database_path: Path,
    retriever: PolicyRetriever,
    llm: LLMClient,
) -> DemoResult:
    """Run product card, review, and support scenarios."""

    product_repository = ProductRepository(database_path)
    review_repository = ReviewRepository(database_path)
    order_repository = OrderRepository(database_path)

    product = product_repository.get_by_sku("LAP-0001")
    if product is None:
        raise ValueError("Demo product LAP-0001 was not found.")

    reviews = review_repository.get_by_sku(product.sku)
    if not reviews:
        raise ValueError("Demo product has no reviews.")

    trace_event("demo_content_started", {"sku": product.sku})
    content = run_content_pipeline(product, llm)
    trace_event(
        "demo_content_completed",
        {"sku": product.sku, "status": content.status},
    )
    taxonomy = load_defect_taxonomy(_project_root() / "data/taxonomy/laptop_defects.yaml")
    trace_event("demo_review_analysis_started", {"review_id": reviews[0].review_id})
    review = classify_review(reviews[0], taxonomy, llm)
    trace_event(
        "demo_review_analysis_completed",
        {"review_id": reviews[0].review_id, "defect_id": review.defect_id},
    )

    registry = ToolRegistry(
        [
            GetProductTool(product_repository),
            GetOrderTool(order_repository),
            SearchPolicyTool(retriever),
        ]
    )
    trace_event("demo_support_started", {})
    support = SupportAgent(registry, llm).run(
        message="Сколько дней можно вернуть товар?",
        session_id="demo-session",
        history=[],
    )
    trace_event(
        "demo_support_completed",
        {"status": support.status, "citations": support.citations},
    )

    return DemoResult(
        product_sku=product.sku,
        content=content,
        review=review,
        support=support,
    )

def format_demo_result(result: DemoResult) -> str:
    """Format demo results for terminal output."""

    title = (
        result.content.content.title
        if result.content.content is not None
        else "—"
    )
    severity = result.review.severity or "—"
    citations = ", ".join(result.support.citations) or "—"

    return (
        "=== Карточка ===\n"
        f"SKU: {result.product_sku}\n"
        f"Статус: {result.content.status}\n"
        f"Заголовок: {title}\n\n"
        "=== Отзыв ===\n"
        f"Дефект: {result.review.defect_id}\n"
        f"Серьёзность: {severity}\n\n"
        "=== Поддержка ===\n"
        f"Статус: {result.support.status}\n"
        f"Ответ: {result.support.text}\n"
        f"Цитаты: {citations}"
    )

def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]
