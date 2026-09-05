from marketplace_agent.batch.checkpoints import BatchCheckpointStore
from marketplace_agent.domain.models import PipelineResult


def test_checkpoint_store_returns_saved_successful_result(
    tmp_path,
) -> None:
    store = BatchCheckpointStore(tmp_path / "batch-checkpoints.db")
    result = PipelineResult(
        sku="LAP-001",
        attempts=1,
        status="completed",
    )

    store.save_success("run-001", result)

    assert store.load_successes("run-001") == {"LAP-001": result}

def test_checkpoint_store_loads_manual_review_items(
    tmp_path,
) -> None:
    store = BatchCheckpointStore(tmp_path / "batch-checkpoints.db")
    result = PipelineResult(
        sku="LAP-001",
        attempts=3,
        status="manual_review",
    )

    store.save_manual_review("run-001", result, "validation failed")

    assert store.load_manual_review("run-001") == {
        "LAP-001": "validation failed"
    }