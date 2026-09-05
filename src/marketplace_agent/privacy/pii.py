import gc
import re
from typing import Any, Self

import spacy

PHONE_PATTERN = re.compile(
    r"(?<!\d)(?:\+7|8)[\s().-]*\d{3}[\s().-]*\d{3}"
    r"[\s.-]*\d{2}[\s.-]*\d{2}(?!\d)"
)
EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
ORDER_NUMBER_PATTERN = re.compile(
    r"(?<![A-Z0-9-])ORD-\d{6}(?!\d)|№\s*\d{4,}",
    re.IGNORECASE,
)
ADDRESS_PATTERN = re.compile(
    r"\b(?:ул\.?|улица|проспект)\s+[А-ЯЁ][а-яё-]+"
    r"(?:\s+[А-ЯЁ][а-яё-]+)?\s*,\s*"
    r"(?:д\.?|дом)\s*\d+[а-яА-Я]?"
    r"(?:\s*,\s*(?:кв\.?|квартира)\s*\d+)?",
    re.IGNORECASE,
)


class PiiRedactor:
    """Detect and redact PII with patterns and a local NER model."""

    def __init__(self) -> None:
        self._nlp: Any | None = None

    def __enter__(self) -> Self:
        self._load_model()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def detect_types(self, text: str) -> set[str]:
        """Return PII types detected in text."""

        if not self._load_model():
            return set()

        detected_types: set[str] = set()
        if PHONE_PATTERN.search(text):
            detected_types.add("phone")
        if EMAIL_PATTERN.search(text):
            detected_types.add("email")
        if ORDER_NUMBER_PATTERN.search(text):
            detected_types.add("order")
        if ADDRESS_PATTERN.search(text):
            detected_types.add("address")
        ner_text = ADDRESS_PATTERN.sub("[ADDRESS]", text)
        if any(
            entity.label_ in {"PERSON", "PER"}
            for entity in self._nlp(ner_text).ents
        ):
            detected_types.add("person")
        return detected_types

    def redact(self, text: str) -> str:
        """Replace detected PII with safe placeholders."""

        if not self._load_model():
            return "[REDACTION_UNAVAILABLE]"

        redacted = PHONE_PATTERN.sub("[PHONE]", text)
        redacted = EMAIL_PATTERN.sub("[EMAIL]", redacted)
        redacted = ORDER_NUMBER_PATTERN.sub("[ORDER]", redacted)
        redacted = ADDRESS_PATTERN.sub("[ADDRESS]", redacted)

        person_entities = [
            entity
            for entity in self._nlp(redacted).ents
            if entity.label_ in {"PERSON", "PER"}
        ]
        for entity in reversed(person_entities):
            redacted = (
                f"{redacted[:entity.start_char]}[PERSON]"
                f"{redacted[entity.end_char:]}"
            )
        return redacted

    def close(self) -> None:
        """Release the loaded model after one operation."""

        self._nlp = None
        gc.collect()

    def _load_model(self) -> bool:
        if self._nlp is not None:
            return True

        try:
            self._nlp = spacy.load(
                "ru_core_news_sm",
                disable=[
                    "morphologizer",
                    "parser",
                    "attribute_ruler",
                    "lemmatizer",
                ],
            )
        except (ImportError, OSError):
            return False

        return True


def detect_pii_types(text: str) -> set[str]:
    """Return types of personal or order data found in text."""

    with PiiRedactor() as redactor:
        return redactor.detect_types(text)


def redact_pii(text: str) -> str:
    """Replace personal and order data with safe placeholders."""

    with PiiRedactor() as redactor:
        return redactor.redact(text)