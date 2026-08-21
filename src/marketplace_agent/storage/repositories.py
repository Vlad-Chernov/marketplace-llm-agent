import json
import sqlite3
from pathlib import Path

from marketplace_agent.domain.models import Order, Product, Review


class ProductRepository:
    """Persist and retrieve product models in SQLite."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def save_many(self, products: list[Product]) -> None:
        """Insert or update a list of products."""

        rows = [
            (
                product.sku,
                product.category,
                product.brand,
                product.model,
                str(product.price),
                product.sales_count,
                product.supplier_description,
                json.dumps(product.attributes, ensure_ascii=False),
                json.dumps(product.true_attributes, ensure_ascii=False),
            )
            for product in products
        ]

        with sqlite3.connect(self.database_path) as connection:
            connection.executemany(
                """
                INSERT INTO products (
                    sku, category, brand, model, price, sales_count,
                    supplier_description, attributes_json, true_attributes_json
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(sku) DO UPDATE SET
                    category = excluded.category,
                    brand = excluded.brand,
                    model = excluded.model,
                    price = excluded.price,
                    sales_count = excluded.sales_count,
                    supplier_description = excluded.supplier_description,
                    attributes_json = excluded.attributes_json,
                    true_attributes_json = excluded.true_attributes_json
                """,
                rows,
            )

    def get_by_sku(self, sku: str) -> Product | None:
        """Return one product by SKU, or None when it does not exist."""

        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM products WHERE sku = ?",
                (sku,),
            ).fetchone()

        if row is None:
            return None

        return Product(
            sku=row["sku"],
            category=row["category"],
            brand=row["brand"],
            model=row["model"],
            price=row["price"],
            sales_count=row["sales_count"],
            supplier_description=row["supplier_description"],
            attributes=json.loads(row["attributes_json"]),
            true_attributes=json.loads(row["true_attributes_json"]),
        )

class ReviewRepository:
    """Persist and retrieve review models in SQLite."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def save_many(self, reviews: list[Review]) -> None:
        """Insert or update a list of reviews."""

        rows = [
            (
                review.review_id,
                review.sku,
                review.rating,
                review.text,
                review.created_at.isoformat(),
                review.helpful_count,
                review.defect_label,
                int(review.contains_personal_data),
                int(review.is_delivery_review),
            )
            for review in reviews
        ]

        with sqlite3.connect(self.database_path) as connection:
            connection.executemany(
                """
                INSERT INTO reviews (
                    review_id, sku, rating, text, created_at, helpful_count,
                    defect_label, contains_personal_data, is_delivery_review
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(review_id) DO UPDATE SET
                    sku = excluded.sku,
                    rating = excluded.rating,
                    text = excluded.text,
                    created_at = excluded.created_at,
                    helpful_count = excluded.helpful_count,
                    defect_label = excluded.defect_label,
                    contains_personal_data = excluded.contains_personal_data,
                    is_delivery_review = excluded.is_delivery_review
                """,
                rows,
            )

    def get_by_sku(self, sku: str) -> list[Review]:
        """Return all reviews for one product SKU."""

        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM reviews WHERE sku = ? ORDER BY created_at",
                (sku,),
            ).fetchall()

        return [
            Review(
                review_id=row["review_id"],
                sku=row["sku"],
                rating=row["rating"],
                text=row["text"],
                created_at=row["created_at"],
                helpful_count=row["helpful_count"],
                defect_label=row["defect_label"],
                contains_personal_data=bool(row["contains_personal_data"]),
                is_delivery_review=bool(row["is_delivery_review"]),
            )
            for row in rows
        ]


class OrderRepository:
    """Persist and retrieve order models in SQLite."""

    def get_by_id(self, order_id: str) -> Order | None:
        """Return one order by ID, or None when it does not exist."""

        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                "SELECT * FROM orders WHERE order_id = ?",
                (order_id,),
            ).fetchone()

        if row is None:
            return None

        return Order(
            order_id=row["order_id"],
            session_id=row["session_id"],
            sku=row["sku"],
            status=row["status"],
            price=row["price"],
            purchased_at=row["purchased_at"],
            delivered_at=row["delivered_at"],
        )   

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

    def save_many(self, orders: list[Order]) -> None:
        """Insert or update a list of orders."""

        rows = [
            (
                order.order_id,
                order.session_id,
                order.sku,
                order.status,
                str(order.price),
                order.purchased_at.isoformat(),
                order.delivered_at.isoformat()
                if order.delivered_at is not None
                else None,
            )
            for order in orders
        ]

        with sqlite3.connect(self.database_path) as connection:
            connection.executemany(
                """
                INSERT INTO orders (
                    order_id, session_id, sku, status, price,
                    purchased_at, delivered_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(order_id) DO UPDATE SET
                    session_id = excluded.session_id,
                    sku = excluded.sku,
                    status = excluded.status,
                    price = excluded.price,
                    purchased_at = excluded.purchased_at,
                    delivered_at = excluded.delivered_at
                """,
                rows,
            )

    def get_by_session(self, session_id: str) -> list[Order]:
        """Return orders visible to one customer session."""

        with sqlite3.connect(self.database_path) as connection:
            connection.row_factory = sqlite3.Row
            rows = connection.execute(
                "SELECT * FROM orders WHERE session_id = ? ORDER BY purchased_at",
                (session_id,),
            ).fetchall()

        return [
            Order(
                order_id=row["order_id"],
                session_id=row["session_id"],
                sku=row["sku"],
                status=row["status"],
                price=row["price"],
                purchased_at=row["purchased_at"],
                delivered_at=row["delivered_at"],
            )
            for row in rows
        ]