
from collections.abc import Sequence
from decimal import Decimal

from marketplace_agent.catalog.extractor import (
    AttributeExtractionResult,
)
from marketplace_agent.content import pipeline
from marketplace_agent.content.examples import ContentExample
from marketplace_agent.domain.models import (
    GeneratedContent,
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


def make_example() -> ContentExample:
    return ContentExample(
        example_id="EXAMPLE-001",
        category="laptops",
        confirmed_attributes={"ram_gb": "16"},
        content=GeneratedContent(
            title="Ноутбук для работы",
            bullets=["16 GB RAM"],
            description="Карточка-пример.",
            keywords=["ноутбук"],
            used_attributes={"ram_gb": "16"},
        ),
    )


def extraction() -> AttributeExtractionResult:
    return AttributeExtractionResult(
        attributes={},
        unsupported_facts=[],
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

    def build_graph(
        _: FakeLLMClient,
        *,
        examples: Sequence[ContentExample] = (),
    ) -> Graph:
        del examples
        return Graph()

    monkeypatch.setattr(
        pipeline,
        "extract_attributes",
        lambda *_: extraction(),
    )
    monkeypatch.setattr(
        pipeline,
        "build_content_graph",
        build_graph,
    )

    result = pipeline.run_content_pipeline(
        make_product(),
        FakeLLMClient(),
    )

    assert result == expected
    assert received_states[0]["attempts"] == 0
    assert received_states[0]["max_attempts"] == 3


def test_pipeline_passes_examples_to_content_graph(
    monkeypatch,
) -> None:
    received_examples: list[ContentExample] = []
    expected = PipelineResult(
        sku="LAP-0001",
        attempts=1,
        status="completed",
    )

    class Graph:
        def invoke(
            self,
            _: dict[str, object],
        ) -> dict[str, PipelineResult]:
            return {"result": expected}

    def build_graph(
        _: FakeLLMClient,
        *,
        examples: list[ContentExample],
    ) -> Graph:
        received_examples.extend(examples)
        return Graph()

    monkeypatch.setattr(
        pipeline,
        "extract_attributes",
        lambda *_: extraction(),
    )
    monkeypatch.setattr(
        pipeline,
        "build_content_graph",
        build_graph,
    )

    example = make_example()
    result = pipeline.run_content_pipeline(
        make_product(),
        FakeLLMClient(),
        examples=[example],
    )

    assert result == expected
    assert received_examples == [example]