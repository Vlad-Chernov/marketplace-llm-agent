import json
from decimal import Decimal
from pathlib import Path

import pytest

from marketplace_agent.domain.models import (
    GeneratedContent,
    PipelineResult,
    Product,
    RuleViolation,
)
from marketplace_agent.evals.content_pipeline import (
    ContentPipelineCaseResult,
    compare_pipeline_versions,
    run_pipeline_version,
)
from marketplace_agent.llm.base import FakeLLMClient, LLMResponse, Message


def violation(rule_id: str) -> RuleViolation:
    return RuleViolation(
        rule_id=rule_id,
        message="Нарушение.",
        severity="high",
        field="description",
    )


def test_compares_baseline_and_validated_pipeline() -> None:
    baseline_results = [
        ContentPipelineCaseResult(
            sku="LAP-0001",
            true_attributes={"ram_gb": "16", "storage_gb": "512"},
            extracted_attributes={"ram_gb": "16"},
            used_attributes={"ram_gb": "16", "invented_feature": "да"},
            violations=[violation("unsupported-claim")],
            latency_ms=100,
            cost_usd=0.01,
        ),
        ContentPipelineCaseResult(
            sku="LAP-0002",
            true_attributes={"ram_gb": "8"},
            extracted_attributes={"ram_gb": "16"},
            used_attributes={"ram_gb": "16"},
            violations=[],
            latency_ms=200,
            cost_usd=0.02,
        ),
    ]
    validated_results = [
        ContentPipelineCaseResult(
            sku="LAP-0001",
            true_attributes={"ram_gb": "16", "storage_gb": "512"},
            extracted_attributes={"ram_gb": "16", "storage_gb": "512"},
            used_attributes={"ram_gb": "16", "storage_gb": "512"},
            violations=[],
            latency_ms=300,
            cost_usd=0.03,
        ),
        ContentPipelineCaseResult(
            sku="LAP-0002",
            true_attributes={"ram_gb": "8"},
            extracted_attributes={"ram_gb": "8"},
            used_attributes={"ram_gb": "8"},
            violations=[violation("title-length")],
            latency_ms=400,
            cost_usd=0.04,
        ),
    ]

    comparison = compare_pipeline_versions(
        baseline_results,
        validated_results,
    )

    assert comparison.baseline.attribute_f1 == pytest.approx(0.4)
    assert comparison.validated.attribute_f1 == 1.0
    assert comparison.baseline.hallucination_rate == pytest.approx(2 / 3)
    assert comparison.validated.hallucination_rate == 0.0
    assert comparison.baseline.violation_count == 1
    assert comparison.validated.violation_count == 1
    assert comparison.baseline.average_latency_ms == 150
    assert comparison.validated.average_latency_ms == 350
    assert comparison.validated.total_cost_usd == pytest.approx(0.07)
    assert comparison.corrected_skus == ["LAP-0001"]
    assert comparison.degraded_skus == ["LAP-0002"]

def test_calculates_outcomes_from_statuses() -> None:
    results = [
        ContentPipelineCaseResult(
            sku="LAP-0001",
            status="completed",
            attempts=1,
            true_attributes={},
            extracted_attributes={},
            used_attributes={},
            violations=[],
            latency_ms=100,
            cost_usd=0.01,
        ),
        ContentPipelineCaseResult(
            sku="LAP-0002",
            status="manual_review",
            attempts=3,
            true_attributes={},
            extracted_attributes={},
            used_attributes={},
            violations=[violation("title-length")],
            latency_ms=200,
            cost_usd=0.02,
        ),
        ContentPipelineCaseResult(
            sku="LAP-0003",
            status="error",
            attempts=0,
            true_attributes={},
            extracted_attributes={},
            used_attributes={},
            violations=[],
            latency_ms=300,
            cost_usd=0.03,
            error="LLMProviderError: HTTP 429",
        ),
    ]

    comparison = compare_pipeline_versions(results, results)

    assert comparison.baseline.success_rate == pytest.approx(1 / 3)
    assert comparison.baseline.failures_by_type == {
        "LLMProviderError": 1,
        "manual_review": 1,
    }
    assert comparison.baseline.prompt_tokens == 0
    assert comparison.baseline.completion_tokens == 0

def make_product(sku: str) -> Product:
    return Product(
        sku=sku,
        category="laptops",
        brand="Lenovo",
        model="IdeaPad",
        price=Decimal(75000),
        sales_count=10,
        supplier_description="Ноутбук Lenovo.",
        attributes={},
    )


def test_runs_each_sku_with_a_new_client_and_trace() -> None:
    created_clients: list[FakeLLMClient] = []

    def llm_factory() -> FakeLLMClient:
        client = FakeLLMClient()
        created_clients.append(client)
        return client

    def completed_pipeline(
        product: Product,
        _: FakeLLMClient,
        __: int,
    ) -> PipelineResult:
        return PipelineResult(
            sku=product.sku,
            attempts=1,
            status="completed",
        )

    results = run_pipeline_version(
        products=[make_product("LAP-0001"), make_product("LAP-0002")],
        llm_factory=llm_factory,
        pipeline=completed_pipeline,
        max_attempts=3,
        input_price_per_million=0.0,
        output_price_per_million=0.0,
    )

    assert len(created_clients) == 2
    assert [result.status for result in results] == [
        "completed",
        "completed",
    ]
    assert results[0].trace[-1]["event_type"] == "completed"


def test_keeps_completed_generated_content() -> None:
    generated = GeneratedContent(
        title="Ноутбук Lenovo",
        bullets=["16 ГБ RAM"],
        description="Описание.",
        keywords=["ноутбук"],
        used_attributes={"ram_gb": "16"},
    )

    results = run_pipeline_version(
        products=[make_product("LAP-0001")],
        llm_factory=FakeLLMClient,
        pipeline=lambda product, _llm, _limit: PipelineResult(
            sku=product.sku,
            content=generated,
            attempts=1,
            status="completed",
        ),
        max_attempts=3,
        input_price_per_million=0.0,
        output_price_per_million=0.0,
    )

    assert results[0].content == generated


def test_reports_progress_for_each_pipeline_outcome() -> None:
    events: list[dict[str, object]] = []

    def completed_pipeline(
        product: Product,
        _: FakeLLMClient,
        __: int,
    ) -> PipelineResult:
        return PipelineResult(
            sku=product.sku,
            attempts=1,
            status="completed",
        )

    run_pipeline_version(
        products=[make_product("LAP-0001")],
        llm_factory=FakeLLMClient,
        pipeline=completed_pipeline,
        max_attempts=3,
        input_price_per_million=0.0,
        output_price_per_million=0.0,
        version="langgraph-v1",
        progress=events.append,
    )

    assert events == [
        {
            "event_type": "started",
            "version": "langgraph-v1",
            "sku": "LAP-0001",
            "position": 1,
            "total": 1,
        },
        {
            "event_type": "completed",
            "version": "langgraph-v1",
            "sku": "LAP-0001",
            "position": 1,
            "total": 1,
        },
    ]


def test_reports_progress_before_each_llm_call() -> None:
    events: list[dict[str, object]] = []

    def pipeline_that_calls_llm(
        product: Product,
        llm: FakeLLMClient,
        _: int,
    ) -> PipelineResult:
        llm.chat(
            messages=[Message(role="user", content="Технический запрос")],
            tools=None,
            response_schema=None,
            temperature=0.0,
            max_tokens=10,
        )
        return PipelineResult(
            sku=product.sku,
            attempts=1,
            status="completed",
        )

    run_pipeline_version(
        products=[make_product("LAP-0001")],
        llm_factory=lambda: FakeLLMClient(
            [
                LLMResponse(
                    content="ОК",
                    model="fake-model",
                    prompt_tokens=1,
                    completion_tokens=1,
                )
            ]
        ),
        pipeline=pipeline_that_calls_llm,
        max_attempts=3,
        input_price_per_million=0.0,
        output_price_per_million=0.0,
        version="langgraph-v1",
        progress=events.append,
    )

    assert [event["event_type"] for event in events] == [
        "started",
        "llm_call_started",
        "completed",
    ]

def test_content_pipeline_manifest_has_twelve_unique_skus() -> None:
    manifest = json.loads(
        Path("data/gold/content_pipeline_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(manifest["skus"]) == 12
    assert len(set(manifest["skus"])) == 12

def test_treats_status_recovery_as_an_improvement() -> None:
    legacy_results = [
        ContentPipelineCaseResult(
            sku="LAP-0001",
            status="error",
            attempts=0,
            true_attributes={},
            extracted_attributes={},
            used_attributes={},
            violations=[],
            latency_ms=0,
            cost_usd=0.0,
            error="LLMProviderError: HTTP 429",
        )
    ]
    graph_results = [
        ContentPipelineCaseResult(
            sku="LAP-0001",
            status="completed",
            attempts=1,
            true_attributes={},
            extracted_attributes={},
            used_attributes={},
            violations=[],
            latency_ms=0,
            cost_usd=0.0,
        )
    ]

    comparison = compare_pipeline_versions(legacy_results, graph_results)

    assert comparison.corrected_skus == ["LAP-0001"]
    assert comparison.degraded_skus == []
