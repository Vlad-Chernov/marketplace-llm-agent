import re

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


def detect_pii_types(text: str) -> set[str]:
    """Return types of personal or order data found in text."""

    detected_types: set[str] = set()

    if PHONE_PATTERN.search(text):
        detected_types.add("phone")
    if EMAIL_PATTERN.search(text):
        detected_types.add("email")
    if ORDER_NUMBER_PATTERN.search(text):
        detected_types.add("order")

    return detected_types


def redact_pii(text: str) -> str:
    """Replace personal and order data with safe placeholders."""

    redacted = PHONE_PATTERN.sub("[PHONE]", text)
    redacted = EMAIL_PATTERN.sub("[EMAIL]", redacted)
    return ORDER_NUMBER_PATTERN.sub("[ORDER]", redacted)