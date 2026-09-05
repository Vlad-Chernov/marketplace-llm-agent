from datetime import date

from marketplace_agent.domain.models import Review
from marketplace_agent.reviews.cleaning import clean_reviews


def make_review(
    review_id: str,
    text: str,
    sku: str = "LAP-0001",
) -> Review:
    return Review(
        review_id=review_id,
        sku=sku,
        rating=3,
        text=text,
        created_at=date(2026, 1, 1),
        helpful_count=0,
    )


def test_normalizes_html_and_masks_personal_data() -> None:
    result = clean_reviews(
        [
            make_review(
                "REV-000001",
                "<p>  Греется   +7 999 123-45-67 "
                "vlad@example.com заказ №12345 </p>",
            )
        ]
    )

    assert result.reviews[0].text == (
        "Греется [PHONE] [EMAIL] заказ [ORDER]"
    )
    assert result.redacted_review_ids == ["REV-000001"]


def test_discards_useless_review_but_keeps_short_defect() -> None:
    result = clean_reviews(
        [
            make_review("REV-000001", "норм"),
            make_review("REV-000002", "греется"),
        ]
    )

    assert [review.review_id for review in result.reviews] == [
        "REV-000002"
    ]
    assert result.discarded_review_ids == ["REV-000001"]

def test_removes_duplicates_only_inside_one_sku() -> None:
    result = clean_reviews(
        [
            make_review(
                "REV-000001",
                "Ноутбук работает тихо и быстро",
            ),
            make_review(
                "REV-000002",
                "Ноутбук работает тихо и быстро",
            ),
            make_review(
                "REV-000003",
                "Ноутбук работает тихо и быстро!",
            ),
            make_review(
                "REV-000004",
                "Ноутбук работает тихо и быстро",
                sku="LAP-0002",
            ),
        ]
    )

    assert [review.review_id for review in result.reviews] == [
        "REV-000001",
        "REV-000004",
    ]
    assert result.duplicate_review_ids == [
        "REV-000002",
        "REV-000003",
    ]

def test_masks_canonical_order_identifier() -> None:
    result = clean_reviews(
        [
            make_review(
                "REV-000005",
                "Проверьте статус заказа ORD-000001.",
            )
        ]
    )

    assert result.reviews[0].text == (
        "Проверьте статус заказа [ORDER]."
    )
    assert result.redacted_review_ids == ["REV-000005"]

def test_masks_person_name_and_address() -> None:
    result = clean_reviews(
        [
            make_review(
                "REV-000006",
                "Анна Петрова: улица Ленина, дом 10. Товар греется.",
            )
        ]
    )

    assert result.reviews[0].text == (
        "[PERSON]: [ADDRESS]. Товар греется."
    )
    assert result.redacted_review_ids == ["REV-000006"]