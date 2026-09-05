from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from time import monotonic

from marketplace_agent.batch.limiter import (
    RateLimitedLLMClient,
    RateLimiter,
)
from marketplace_agent.content.pipeline import run_content_pipeline
from marketplace_agent.domain.models import (
    PipelineResult,
    Product,
)
from marketplace_agent.llm.base import LLMClient
from marketplace_agent.storage.repositories import ProductRepository

Pipeline = Callable[[Product, LLMClient], PipelineResult]


@dataclass(frozen=True)
class BatchSummary:
    """Summarize one batch-processing run."""

    requested_skus: list[str]
    results: dict[str, PipelineResult]
    errors: dict[str, str]
    elapsed_seconds: float


class BatchProcessor:
    """Process product cards concurrently with one shared rate limit."""

    def __init__(
        self,
        products: ProductRepository,
        llm_factory: Callable[[], LLMClient],
        *,
        max_workers: int = 2,
        requests_per_minute: int = 30,
        pipeline: Pipeline = run_content_pipeline,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least 1.")
        if requests_per_minute < 1:
            raise ValueError(
                "requests_per_minute must be at least 1."
            )

        self._products = products
        self._llm_factory = llm_factory
        self._max_workers = max_workers
        self._requests_per_minute = requests_per_minute
        self._pipeline = pipeline

    def run(
        self,
        product_ids: list[str],
        resume: bool = True,
    ) -> BatchSummary:
        """Run the content pipeline once for every requested SKU."""

        del resume

        if not product_ids:
            raise ValueError("product_ids must not be empty.")
        if len(set(product_ids)) != len(product_ids):
            raise ValueError("product_ids must be unique.")

        started_at = monotonic()
        limiter = RateLimiter(self._requests_per_minute)
        futures: dict[str, Future[PipelineResult]] = {}

        with ThreadPoolExecutor(
            max_workers=self._max_workers
        ) as executor:
            for sku in product_ids:
                futures[sku] = executor.submit(
                    self._process_product,
                    sku,
                    limiter,
                )

        results: dict[str, PipelineResult] = {}
        errors: dict[str, str] = {}

        for sku in product_ids:
            try:
                results[sku] = futures[sku].result()
            except Exception as error:  # noqa: BLE001
                errors[sku] = str(error) or type(error).__name__

        return BatchSummary(
            requested_skus=product_ids,
            results=results,
            errors=errors,
            elapsed_seconds=monotonic() - started_at,
        )

    def _process_product(
        self,
        sku: str,
        limiter: RateLimiter,
    ) -> PipelineResult:
        product = self._products.get_by_sku(sku)
        if product is None:
            raise LookupError("Product not found.")

        client = RateLimitedLLMClient(
            self._llm_factory(),
            limiter,
        )
        return self._pipeline(product, client)