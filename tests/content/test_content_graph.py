from decimal import Decimal

from marketplace_agent.catalog.extractor import (
    AttributeExtractionResult,
)
from marketplace_agent.content import pipeline
from marketplace_agent.domain.models import (
    PipelineResult,
    Product,
)
from marketplace_agent.llm.base import FakeLLMClient


def make_product() -> Product:
    return Product(
        sku="LAP-0001",
        category="laptops",
        brand="Lenovo",
        model="IdeaPad 1000",
        price=Decimal(75000),
        sales_count=10,
        supplier_description="Ноутбук Lenovo с экраном 14 дюймов.",
        attributes={},
    )


def test_pipeline_delegates_content_cycle_to_graph(
    monkeypatch,
) -> None:
    expected = PipelineResult(
        sku="LAP-0001",
        attempts=1,
        status="completed",
    )
    received_states: list[dict[str, object]] = []

    class Graph:
        def invoke(
            self,
            state: dict[str, object],
        ) -> dict[str, PipelineResult]:
            received_states.append(state)
            return {"result": expected}

    def build_graph(_: FakeLLMClient) -> Graph:
        return Graph()

    monkeypatch.setattr(
        pipeline,
        "extract_attributes",
        lambda *_: AttributeExtractionResult(
            attributes={},
            unsupported_facts=[],
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "build_content_graph",
        build_graph,
        raising=False,
    )

    result = pipeline.run_content_pipeline(
        make_product(),
        FakeLLMClient(),
    )

    assert result == expected
    assert received_states[0]["attempts"] == 0
    assert received_states[0]["max_attempts"] == 3