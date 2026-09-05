from decimal import Decimal
from threading import Event, Lock, Thread

import pytest

from marketplace_agent.batch.checkpoints import BatchCheckpointStore
from marketplace_agent.batch.processor import (
    BatchProcessor,
    BatchSummary,
)
from marketplace_agent.domain.models import PipelineResult, Product
from marketplace_agent.llm.base import FakeLLMClient


class FakeProducts:
    def __init__(self, products: dict[str, Product]) -> None:
        self._products = products

    def get_by_sku(self, sku: str) -> Product | None:
        return self._products.get(sku)


def make_product(sku: str) -> Product:
    return Product(
        sku=sku,
        category="laptops",
        brand="Lenovo",
        model="IdeaPad",
        price=Decimal(75000),
        sales_count=10,
        supplier_description="Ноутбук с экраном 14 дюймов.",
    )


def process_product(
    product: Product,
    _: FakeLLMClient,
) -> PipelineResult:
    if product.sku == "LAP-001":
        raise RuntimeError("processing failed")

    return PipelineResult(
        sku=product.sku,
        attempts=1,
        status="completed",
    )


def test_batch_continues_after_one_product_fails() -> None:
    processor = BatchProcessor(
        products=FakeProducts(
            {
                "LAP-001": make_product("LAP-001"),
                "LAP-002": make_product("LAP-002"),
            }
        ),
        llm_factory=FakeLLMClient,
        pipeline=process_product,
    )

    summary = processor.run(["LAP-001", "LAP-002"])

    assert list(summary.results) == ["LAP-002"]
    assert "LAP-001" in summary.errors


def test_batch_runs_two_workers_concurrently() -> None:
    active = 0
    maximum_active = 0
    lock = Lock()
    two_workers_started = Event()
    release_workers = Event()
    summary: BatchSummary | None = None

    def slow_process(
        product: Product,
        _: FakeLLMClient,
    ) -> PipelineResult:
        nonlocal active, maximum_active

        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
            if active == 2:
                two_workers_started.set()

        assert two_workers_started.wait(timeout=1)
        assert release_workers.wait(timeout=1)

        with lock:
            active -= 1

        return PipelineResult(
            sku=product.sku,
            attempts=1,
            status="completed",
        )

    processor = BatchProcessor(
        products=FakeProducts(
            {
                "LAP-001": make_product("LAP-001"),
                "LAP-002": make_product("LAP-002"),
                "LAP-003": make_product("LAP-003"),
            }
        ),
        llm_factory=FakeLLMClient,
        max_workers=2,
        pipeline=slow_process,
    )

    def run_batch() -> None:
        nonlocal summary
        summary = processor.run(
            ["LAP-001", "LAP-002", "LAP-003"]
        )

    thread = Thread(target=run_batch)
    thread.start()

    assert two_workers_started.wait(timeout=1)
    with lock:
        assert active == 2

    release_workers.set()
    thread.join(timeout=1)

    assert not thread.is_alive()
    assert summary is not None
    assert len(summary.results) == 3
    assert maximum_active == 2

def test_batch_rejects_duplicate_skus() -> None:
    processor = BatchProcessor(
        products=FakeProducts({}),
        llm_factory=FakeLLMClient,
        pipeline=process_product,
    )

    with pytest.raises(ValueError, match="unique"):
        processor.run(["LAP-001", "LAP-001"])


def test_batch_records_unknown_sku_and_processes_known_one() -> None:
    processor = BatchProcessor(
        products=FakeProducts(
            {
                "LAP-002": make_product("LAP-002"),
            }
        ),
        llm_factory=FakeLLMClient,
        pipeline=process_product,
    )

    summary = processor.run(["MISSING-001", "LAP-002"])

    assert list(summary.results) == ["LAP-002"]
    assert summary.errors == {
        "MISSING-001": "Product not found.",
    }

def test_batch_resume_skips_previously_successful_sku(
    tmp_path,
) -> None:
    checkpoint_store = BatchCheckpointStore(
        tmp_path / "batch-checkpoints.db"
    )
    products = FakeProducts(
        {
            "LAP-002": make_product("LAP-002"),
        }
    )

    first_processor = BatchProcessor(
        products=products,
        llm_factory=FakeLLMClient,
        checkpoint_store=checkpoint_store,
        pipeline=process_product,
    )
    first_processor.run(["LAP-002"])

    calls: list[str] = []

    def must_not_run(
        product: Product,
        _: FakeLLMClient,
    ) -> PipelineResult:
        calls.append(product.sku)
        raise AssertionError("successful SKU must be skipped")

    resumed_processor = BatchProcessor(
        products=products,
        llm_factory=FakeLLMClient,
        checkpoint_store=checkpoint_store,
        pipeline=must_not_run,
    )

    summary = resumed_processor.run(["LAP-002"], resume=True)

    assert list(summary.results) == ["LAP-002"]
    assert calls == []

def test_batch_sends_manual_review_result_to_queue(
    tmp_path,
) -> None:
    store = BatchCheckpointStore(tmp_path / "batch-checkpoints.db")

    def needs_review(
        product: Product,
        _: FakeLLMClient,
    ) -> PipelineResult:
        return PipelineResult(
            sku=product.sku,
            attempts=3,
            status="manual_review",
        )

    processor = BatchProcessor(
        products=FakeProducts({"LAP-002": make_product("LAP-002")}),
        llm_factory=FakeLLMClient,
        checkpoint_store=store,
        pipeline=needs_review,
    )

    summary = processor.run(["LAP-002"])

    assert summary.results["LAP-002"].status == "manual_review"
    assert store.load_successes(summary.run_id) == {}
    assert store.load_manual_review(summary.run_id) == {
        "LAP-002": "manual_review"
    }

def test_batch_sends_processing_error_to_manual_review_queue(
    tmp_path,
) -> None:
    store = BatchCheckpointStore(tmp_path / "batch-checkpoints.db")
    processor = BatchProcessor(
        products=FakeProducts({"LAP-001": make_product("LAP-001")}),
        llm_factory=FakeLLMClient,
        checkpoint_store=store,
        pipeline=process_product,
    )

    summary = processor.run(["LAP-001"])

    assert summary.errors == {"LAP-001": "processing failed"}
    assert store.load_manual_review(summary.run_id) == {
        "LAP-001": "processing failed",
    }