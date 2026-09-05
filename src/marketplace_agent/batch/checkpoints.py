
import sqlite3
from pathlib import Path

from marketplace_agent.domain.models import PipelineResult


class BatchCheckpointStore:
    """Persist successful batch items for resume."""

    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS batch_successes (
                    run_id TEXT NOT NULL,
                    sku TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    PRIMARY KEY (run_id, sku)
                )
                """
            )

    def save_success(
        self,
        run_id: str,
        result: PipelineResult,
    ) -> None:
        """Save one completed product result."""

        with sqlite3.connect(self._database_path) as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO batch_successes (
                    run_id, sku, result_json
                ) VALUES (?, ?, ?)
                """,
                (
                    run_id,
                    result.sku,
                    result.model_dump_json(),
                ),
            )

    def load_successes(
        self,
        run_id: str,
    ) -> dict[str, PipelineResult]:
        """Load completed products for one batch run."""

        with sqlite3.connect(self._database_path) as connection:
            rows = connection.execute(
                """
                SELECT sku, result_json
                FROM batch_successes
                WHERE run_id = ?
                ORDER BY rowid
                """,
                (run_id,),
            ).fetchall()

        return {
            sku: PipelineResult.model_validate_json(result_json)
            for sku, result_json in rows
        }