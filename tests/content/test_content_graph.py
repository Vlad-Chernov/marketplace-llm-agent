
import json
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path

from marketplace_agent.catalog.extractor import (
    AttributeExtractionResult,
    ExtractedAttribute,
)
from marketplace_agent.content import graph as content_graph
from marketplace_agent.content import pipeline
from marketplace_agent.content.examples import ContentExample
from marketplace_agent.content.graph import build_content_graph
from marketplace_agent.domain.models import (
    GeneratedContent,
    PipelineResult,
    Product,
)
from marketplace_agent.llm.base import FakeLLMClient
from marketplace_agent.llm.telemetry import TraceWriter, trace_run
from marketplace_agent.validation.grounding import GroundingEvidence


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

def test_graph_records_selected_example_ids(
    monkeypatch,
) -> None:
    example = make_example()
    extracted_attributes = AttributeExtractionResult(
        attributes={
            "ram_gb": ExtractedAttribute(
                value="16",
                confidence=0.95,
            )
        },
        unsupported_facts=[],
    )

    monkeypatch.setattr(
        content_graph,
        "generate_content",
        lambda *_args, **_kwargs: example.content,
    )
    monkeypatch.setattr(
        content_graph,
        "_validate_content",
        lambda *_args: [],
    )

    graph = build_content_graph(
        FakeLLMClient(),
        examples=[example],
    )
    result = graph.invoke(
        {
            "product": make_product(),
            "extracted_attributes": extracted_attributes,
            "evidence": GroundingEvidence(
                confirmed_attributes={"ram_gb": "16"},
                supplier_description=make_product().supplier_description,
            ),
            "deterministic_rules": [],
            "semantic_rules": [],
            "violations": [],
            "attempts": 0,
            "max_attempts": 3,
            "seen_contents": set(),
        }
    )

    assert result["result"].selected_example_ids == ["EXAMPLE-001"]


def test_graph_records_generate_validate_and_completion_steps(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        content_graph,
        "generate_content",
        lambda *_args, **_kwargs: make_example().content,
    )
    monkeypatch.setattr(
        content_graph,
        "_validate_content",
        lambda *_args: [],
    )
    graph = build_content_graph(FakeLLMClient())
    trace_path = tmp_path / "trace.jsonl"

    with trace_run(TraceWriter(trace_path), "run-content"):
        graph.invoke(
            {
                "product": make_product(),
                "extracted_attributes": extraction(),
                "evidence": GroundingEvidence(
                    confirmed_attributes={},
                    supplier_description=make_product().supplier_description,
                ),
                "deterministic_rules": [],
                "semantic_rules": [],
                "violations": [],
                "attempts": 0,
                "max_attempts": 3,
                "seen_contents": set(),
            }
        )

    events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]

    assert [event["event_type"] for event in events] == [
        "run_started",
        "content_generation_started",
        "content_generation_completed",
        "content_validation_started",
        "content_validation_completed",
        "content_pipeline_completed",
        "run_completed",
    ]
