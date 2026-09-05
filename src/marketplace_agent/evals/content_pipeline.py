import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Any

from pydantic import BaseModel

from marketplace_agent.data_generation.catalog import generate_clean_products
from marketplace_agent.data_generation.noise import noise_product
from marketplace_agent.domain.models import (
    PipelineResult,
    Product,
    RuleViolation,
)
from marketplace_agent.evals.attribute_metrics import (
    evaluate_attribute_extraction,
)
from marketplace_agent.llm.base import LLMClient, LLMResponse, Message

Pipeline = Callable[[Product, LLMClient, int], PipelineResult]


def load_content_pipeline_products(
    manifest_path: Path | str,
) -> list[Product]:
    """Build the fixed noisy product subset used by content-pipeline evals."""

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    clean_products = generate_clean_products(
        count=manifest["catalog_size"],
        seed=manifest["catalog_seed"],
    )
    noise_rng = Random(manifest["noise_seed"])
    noisy_products = [
        noise_product(product, noise_rng) for product in clean_products
    ]
    products_by_sku = {product.sku: product for product in noisy_products}

    missing_skus = [
        sku for sku in manifest["skus"] if sku not in products_by_sku
    ]
    if missing_skus:
        missing_text = ", ".join(missing_skus)
        raise ValueError(f"Unknown SKU in content-pipeline manifest: {missing_text}")

    return [products_by_sku[sku] for sku in manifest["skus"]]

@dataclass(frozen=True)
class ContentPipelineCaseResult:
    sku: str
    true_attributes: Mapping[str, Any]
    extracted_attributes: Mapping[str, Any]
    used_attributes: Mapping[str, Any]
    violations: list[RuleViolation]
    latency_ms: int
    cost_usd: float
    status: str = "completed"
    attempts: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str | None = None
    error: str | None = None
    trace: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class PipelineVersionMetrics:
    attribute_f1: float
    hallucination_rate: float
    violation_count: int
    average_latency_ms: float
    total_cost_usd: float
    success_rate: float
    failures_by_type: dict[str, int]
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class ContentPipelineComparison:
    baseline: PipelineVersionMetrics
    validated: PipelineVersionMetrics
    corrected_skus: list[str]
    degraded_skus: list[str]


def compare_pipeline_versions(
    baseline_results: Sequence[ContentPipelineCaseResult],
    validated_results: Sequence[ContentPipelineCaseResult],
) -> ContentPipelineComparison:
    baseline_skus = [result.sku for result in baseline_results]
    validated_skus = [result.sku for result in validated_results]

    if baseline_skus != validated_skus:
        raise ValueError("Baseline and validated results must use the same SKUs.")

    return ContentPipelineComparison(
        baseline=_calculate_version_metrics(baseline_results),
        validated=_calculate_version_metrics(validated_results),
        corrected_skus=_changed_skus(
            baseline_results,
            validated_results,
            improved=True,
        ),
        degraded_skus=_changed_skus(
            baseline_results,
            validated_results,
            improved=False,
        ),
    )


def _calculate_version_metrics(
    results: Sequence[ContentPipelineCaseResult],
) -> PipelineVersionMetrics:
    if not results:
        raise ValueError("At least one content-pipeline result is required.")

    attribute_metrics = evaluate_attribute_extraction(
        [result.extracted_attributes for result in results],
        [result.true_attributes for result in results],
    )
    used_attribute_count = sum(
        len(result.used_attributes) for result in results
    )
    unsupported_used_attribute_count = sum(
        _unsupported_used_attribute_count(result) for result in results
    )

    failures_by_type: dict[str, int] = {}

    for result in results:
        if result.status == "manual_review":
            failures_by_type["manual_review"] = (
                failures_by_type.get("manual_review", 0) + 1
            )
        elif result.status == "error":
            error_type = (
                result.error.split(":", maxsplit=1)[0]
                if result.error
                else "unknown_error"
            )
            failures_by_type[error_type] = (
                failures_by_type.get(error_type, 0) + 1
            )

    return PipelineVersionMetrics(
        success_rate=sum(
            result.status == "completed" for result in results
        )
        / len(results),
        failures_by_type=failures_by_type,
        prompt_tokens=sum(result.prompt_tokens for result in results),
        completion_tokens=sum(
            result.completion_tokens for result in results
        ),
        attribute_f1=attribute_metrics.overall.f1,
        hallucination_rate=(
            unsupported_used_attribute_count / used_attribute_count
            if used_attribute_count
            else 0.0
        ),
        violation_count=sum(len(result.violations) for result in results),
        average_latency_ms=(
            sum(result.latency_ms for result in results) / len(results)
        ),
        total_cost_usd=sum(result.cost_usd for result in results),
    )


def _unsupported_used_attribute_count(
    result: ContentPipelineCaseResult,
) -> int:
    return sum(
        key not in result.true_attributes
        or value != result.true_attributes[key]
        for key, value in result.used_attributes.items()
    )


def _changed_skus(
    baseline_results: Sequence[ContentPipelineCaseResult],
    validated_results: Sequence[ContentPipelineCaseResult],
    improved: bool,
) -> list[str]:
    changed_skus: list[str] = []

    for baseline, validated in zip(
        baseline_results,
        validated_results,
        strict=True,
    ):
        status_improved = (
            baseline.status != "completed"
            and validated.status == "completed"
        )
        status_degraded = (
            baseline.status == "completed"
            and validated.status != "completed"
        )
        violations_improved = (
            len(validated.violations) < len(baseline.violations)
        )
        violations_degraded = (
            len(validated.violations) > len(baseline.violations)
        )

        if improved and (status_improved or violations_improved) or not improved and (status_degraded or violations_degraded):
            changed_skus.append(baseline.sku)

    return changed_skus

class MeteredLLMClient:
    """Wrap an LLM client and collect usage for one product run."""

    def __init__(
        self,
        client: LLMClient,
        input_price_per_million: float,
        output_price_per_million: float,
    ) -> None:
        self._client = client
        self._input_price_per_million = input_price_per_million
        self._output_price_per_million = output_price_per_million
        self.latency_ms = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.model: str | None = None

    @property
    def cost_usd(self) -> float:
        return (
            self.prompt_tokens / 1_000_000 * self._input_price_per_million
            + self.completion_tokens
            / 1_000_000
            * self._output_price_per_million
        )

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        response_schema: type[BaseModel] | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        started_at = perf_counter()
        response = self._client.chat(
            messages=messages,
            tools=tools,
            response_schema=response_schema,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.latency_ms += round((perf_counter() - started_at) * 1000)
        self.prompt_tokens += response.prompt_tokens
        self.completion_tokens += response.completion_tokens
        self.model = response.model
        return response

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed(texts)


def run_pipeline_version(
    products: Sequence[Product],
    llm_factory: Callable[[], LLMClient],
    pipeline: Pipeline,
    max_attempts: int,
    input_price_per_million: float,
    output_price_per_million: float,
) -> list[ContentPipelineCaseResult]:
    """Run one pipeline version for a fixed product subset."""

    results: list[ContentPipelineCaseResult] = []

    for product in products:
        metered_llm = MeteredLLMClient(
            llm_factory(),
            input_price_per_million,
            output_price_per_million,
        )
        trace: list[dict[str, object]] = [
            {"event_type": "started", "sku": product.sku}
        ]

        try:
            pipeline_result = pipeline(
                product,
                metered_llm,
                max_attempts,
            )
        except Exception as error:  # noqa: BLE001
            error_text = f"{type(error).__name__}: {error}"
            trace.append(
                {
                    "event_type": "error",
                    "error_type": type(error).__name__,
                }
            )
            results.append(
                ContentPipelineCaseResult(
                    sku=product.sku,
                    status="error",
                    attempts=0,
                    true_attributes=product.true_attributes,
                    extracted_attributes={},
                    used_attributes={},
                    violations=[],
                    latency_ms=metered_llm.latency_ms,
                    cost_usd=metered_llm.cost_usd,
                    prompt_tokens=metered_llm.prompt_tokens,
                    completion_tokens=metered_llm.completion_tokens,
                    model=metered_llm.model,
                    error=error_text,
                    trace=trace,
                )
            )
            continue

        content = pipeline_result.content
        used_attributes = content.used_attributes if content is not None else {}
        trace.append(
            {
                "event_type": pipeline_result.status,
                "attempts": pipeline_result.attempts,
                "violation_rule_ids": [
                    violation.rule_id for violation in pipeline_result.violations
                ],
            }
        )

        results.append(
            ContentPipelineCaseResult(
                sku=product.sku,
                status=pipeline_result.status,
                attempts=pipeline_result.attempts,
                true_attributes=product.true_attributes,
                extracted_attributes=used_attributes,
                used_attributes=used_attributes,
                violations=pipeline_result.violations,
                latency_ms=metered_llm.latency_ms,
                cost_usd=metered_llm.cost_usd,
                prompt_tokens=metered_llm.prompt_tokens,
                completion_tokens=metered_llm.completion_tokens,
                model=metered_llm.model,
                trace=trace,
            )
        )

    return results
