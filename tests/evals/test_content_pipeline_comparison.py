import pytest

from marketplace_agent.domain.models import RuleViolation
from marketplace_agent.evals.content_pipeline import (
    ContentPipelineCaseResult,
    compare_pipeline_versions,
)


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