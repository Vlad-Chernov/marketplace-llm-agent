import sqlite3
from pathlib import Path

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS products (
    sku TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    brand TEXT NOT NULL,
    model TEXT NOT NULL,
    price TEXT NOT NULL,
    sales_count INTEGER NOT NULL,
    supplier_description TEXT NOT NULL,
    attributes_json TEXT NOT NULL,
    true_attributes_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    review_id TEXT PRIMARY KEY,
    sku TEXT NOT NULL,
    rating INTEGER NOT NULL,
    text TEXT NOT NULL,
    created_at TEXT NOT NULL,
    helpful_count INTEGER NOT NULL,
    defect_label TEXT,
    contains_personal_data INTEGER NOT NULL,
    is_delivery_review INTEGER NOT NULL,
    FOREIGN KEY (sku) REFERENCES products(sku)
);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    sku TEXT NOT NULL,
    status TEXT NOT NULL,
    price TEXT NOT NULL,
    purchased_at TEXT NOT NULL,
    delivered_at TEXT,
    FOREIGN KEY (sku) REFERENCES products(sku)
);

CREATE TABLE IF NOT EXISTS processing_results (
    result_id INTEGER PRIMARY KEY AUTOINCREMENT,
    sku TEXT NOT NULL,
    status TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (sku) REFERENCES products(sku)
);
"""


def initialize_database(database_path: Path) -> None:
    """Create the SQLite database and all project tables."""

    database_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(database_path) as connection:
        connection.executescript(SCHEMA)