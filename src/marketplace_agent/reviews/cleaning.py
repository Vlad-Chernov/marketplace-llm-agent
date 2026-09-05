import html
import re
from difflib import SequenceMatcher

from pydantic import BaseModel

from marketplace_agent.domain.models import Review
from marketplace_agent.privacy.pii import PiiRedactor

HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")

USELESS_TEXTS = {"норм", "ок", "спасибо", "хорошо"}
DEFECT_MARKERS = (
    "греется",
    "перегрев",
    "мерцает",
    "клавиш",
    "wifi",
    "wi-fi",
    "батаре",
)


class ReviewCleaningResult(BaseModel):
    """Store cleaned reviews and audit IDs."""

    reviews: list[Review]
    duplicate_review_ids: list[str]
    discarded_review_ids: list[str]
    redacted_review_ids: list[str]


def clean_reviews(reviews: list[Review]) -> ReviewCleaningResult:
    """Normalize reviews, mask PII and discard obvious noise."""

    cleaned_reviews: list[Review] = []
    discarded_review_ids: list[str] = []
    redacted_review_ids: list[str] = []
    duplicate_review_ids: list[str] = []
    kept_texts_by_sku: dict[str, list[str]] = {}

    redactor = PiiRedactor()
    try:
        for review in reviews:
            cleaned_text, was_redacted = _clean_text(
                review.text,
                redactor,
            )

            if _is_useless(cleaned_text):
                discarded_review_ids.append(review.review_id)
                continue

            if was_redacted:
                redacted_review_ids.append(review.review_id)

            kept_texts = kept_texts_by_sku.setdefault(review.sku, [])
            if _is_duplicate(cleaned_text, kept_texts):
                duplicate_review_ids.append(review.review_id)
                continue

            kept_texts.append(cleaned_text)
            cleaned_reviews.append(review.model_copy(update={"text": cleaned_text}))

    finally:
        redactor.close()

    return ReviewCleaningResult(
        reviews=cleaned_reviews,
        duplicate_review_ids=duplicate_review_ids,
        discarded_review_ids=discarded_review_ids,
        redacted_review_ids=redacted_review_ids,
    )


def _clean_text(
    text: str,
    redactor: PiiRedactor,
) -> tuple[str, bool]:
    normalized_text = WHITESPACE_PATTERN.sub(
        " ",
        HTML_TAG_PATTERN.sub(" ", html.unescape(text)),
    ).strip()
    redacted_text = redactor.redact(normalized_text)
    return redacted_text, redacted_text != normalized_text


def _is_useless(text: str) -> bool:
    normalized_text = text.casefold().strip(" .!?,")
    has_defect_marker = any(
        marker in normalized_text for marker in DEFECT_MARKERS
    )
    return normalized_text in USELESS_TEXTS and not has_defect_marker

def _is_duplicate(text: str, kept_texts: list[str]) -> bool:
    normalized_text = text.casefold()
    return any(
        SequenceMatcher(
            None,
            normalized_text,
            kept_text.casefold(),
        ).ratio()
        >= 0.92
        for kept_text in kept_texts
    )