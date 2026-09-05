from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from hashlib import sha256
from time import monotonic

from marketplace_agent.batch.checkpoints import BatchCheckpointStore
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

    run_id: str
    requested_skus: list[str]
    results: dict[str, PipelineResult]
    errors: dict[str, str]
    elapsed_seconds: float
    throughput_per_second: float


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
        checkpoint_store: BatchCheckpointStore | None = None,
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
        self._checkpoint_store = checkpoint_store

    def run(
        self,
        product_ids: list[str],
        resume: bool = True,
    ) -> BatchSummary:
        """Run the content pipeline once for every requested SKU."""

        run_id = sha256(
            "\n".join(product_ids).encode("utf-8")
        ).hexdigest()
        successes = (
            self._checkpoint_store.load_successes(run_id)
            if resume and self._checkpoint_store is not None
            else {}
        )
        pending_skus = [
            sku for sku in product_ids if sku not in successes
        ]

        if not product_ids:
            raise ValueError("product_ids must not be empty.")
        if len(set(product_ids)) != len(product_ids):
            raise ValueError("product_ids must be unique.")

        started_at = monotonic()
        limiter = RateLimiter(self._requests_per_minute)
        futures: dict[str, Future[PipelineResult]] = {}
        results = successes.copy()

        with ThreadPoolExecutor(
            max_workers=self._max_workers
        ) as executor:
            for sku in pending_skus:
                futures[sku] = executor.submit(
                    self._process_product,
                    sku,
                    limiter,
                )

        errors: dict[str, str] = {}

        for sku in pending_skus:
            try:
                result = futures[sku].result()
                results[sku] = result

                if (
                    self._checkpoint_store is not None
                    and result.status == "completed"
                ):
                    self._checkpoint_store.save_success(
                        run_id,
                        result,
                    )
                elif self._checkpoint_store is not None:
                    self._checkpoint_store.save_manual_review(
                        run_id,
                        result,
                        result.status,
                    )
            except Exception as error:  # noqa: BLE001
                errors[sku] = str(error) or type(error).__name__
                if self._checkpoint_store is not None:
                    self._checkpoint_store.save_manual_review(
                        run_id,
                        PipelineResult(
                            sku=sku,
                            attempts=0,
                            status="manual_review",
                        ),
                        errors[sku],
                    )


        elapsed_seconds = monotonic() - started_at
        completed_count = sum(
            result.status == "completed"
            for result in results.values()
        )
        return BatchSummary(
            run_id=run_id,
            requested_skus=product_ids,
            results=results,
            errors=errors,
            elapsed_seconds=elapsed_seconds,
            throughput_per_second=(
                completed_count / elapsed_seconds
                if elapsed_seconds > 0
                else 0.0
            ),
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