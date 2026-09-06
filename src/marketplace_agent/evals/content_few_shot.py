from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from marketplace_agent.content.examples import ContentExample
from marketplace_agent.content.pipeline import run_content_pipeline
from marketplace_agent.domain.models import PipelineResult, Product
from marketplace_agent.evals.content_pipeline import (
    ContentPipelineCaseResult,
    ContentPipelineComparison,
    compare_pipeline_versions,
    run_pipeline_version,
)
from marketplace_agent.llm.base import LLMClient


class FewShotPipeline(Protocol):
    """Run one product with optional few-shot examples."""

    def __call__(
        self,
        product: Product,
        llm: LLMClient,
        max_attempts: int = 3,
        *,
        examples: Sequence[ContentExample] = (),
    ) -> PipelineResult:
        """Return one pipeline result."""


@dataclass(frozen=True)
class FewShotExperimentResult:
    """Store raw results and aggregate metrics for one comparison."""

    comparison: ContentPipelineComparison
    zero_shot_results: list[ContentPipelineCaseResult]
    few_shot_results: list[ContentPipelineCaseResult]

def serialize_case_result(
    result: ContentPipelineCaseResult,
) -> dict[str, object]:
    """Return fields that are safe to include in an experiment report."""

    return {
        "sku": result.sku,
        "status": result.status,
        "attempts": result.attempts,
        "violations": [
            violation.model_dump()
            for violation in result.violations
        ],
        "latency_ms": result.latency_ms,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "cost_usd": result.cost_usd,
        "model": result.model,
        "error": result.error,
        "trace": result.trace,
        "selected_example_ids": result.selected_example_ids,
    }


def run_few_shot_comparison(
    products: Sequence[Product],
    llm_factory: Callable[[], LLMClient],
    examples: Sequence[ContentExample],
    max_attempts: int,
    input_price_per_million: float,
    output_price_per_million: float,
    pipeline: FewShotPipeline = run_content_pipeline,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> FewShotExperimentResult:
    """Run zero-shot and few-shot pipeline versions on the same products."""

    zero_shot_results = run_pipeline_version(
        products=products,
        llm_factory=llm_factory,
        pipeline=lambda product, llm, attempts: pipeline(
            product,
            llm,
            attempts,
            examples=(),
        ),
        max_attempts=max_attempts,
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
        version="zero-shot-v1",
        progress=progress,
    )
    few_shot_results = run_pipeline_version(
        products=products,
        llm_factory=llm_factory,
        pipeline=lambda product, llm, attempts: pipeline(
            product,
            llm,
            attempts,
            examples=examples,
        ),
        max_attempts=max_attempts,
        input_price_per_million=input_price_per_million,
        output_price_per_million=output_price_per_million,
        version="few-shot-v1",
        progress=progress,
    )

    return FewShotExperimentResult(
        comparison=compare_pipeline_versions(
            zero_shot_results,
            few_shot_results,
        ),
        zero_shot_results=zero_shot_results,
        few_shot_results=few_shot_results,
    )