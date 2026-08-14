import sqlite3
from pathlib import Path

from marketplace_agent.storage.database import initialize_database


def test_initializes_database_with_required_tables(tmp_path: Path) -> None:
    database_path = tmp_path / "marketplace.db"

    initialize_database(database_path)

    with sqlite3.connect(database_path) as connection:
        table_names = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    assert {
        "products",
        "reviews",
        "orders",
        "processing_results",
    } <= table_names