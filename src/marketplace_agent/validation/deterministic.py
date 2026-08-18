import re
from collections.abc import Mapping, Sequence
from typing import Any

from marketplace_agent.domain.models import GeneratedContent, RuleViolation

EMAIL_PATTERN = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
PHONE_PATTERN = re.compile(r"(?:\+7|8)[\s()-]*\d(?:[\s()-]*\d){9}")
URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
EMOJI_PATTERN = re.compile(r"[\U0001F300-\U0001FAFF]")


def validate_deterministic(
    content: GeneratedContent,
    rules: Sequence[Mapping[str, Any]],
) -> list[RuleViolation]:
    violations: list[RuleViolation] = []

    for rule in rules:
        rule_id = rule["id"]

        if rule_id == "title-length":
            violation = _validate_title_length(content, rule)
        elif rule_id == "uppercase-ratio":
            violation = _validate_uppercase_ratio(content, rule)
        elif rule_id == "contact-information":
            violation = _validate_pattern(
                content,
                rule,
                EMAIL_PATTERN,
                PHONE_PATTERN,
            )
        elif rule_id == "external-url":
            violation = _validate_pattern(content, rule, URL_PATTERN)
        elif rule_id == "emoji-spam":
            violation = _validate_emoji_spam(content, rule)
        else:
            violation = None

        if violation is not None:
            violations.append(violation)

    return violations


def _validate_title_length(
    content: GeneratedContent,
    rule: Mapping[str, Any],
) -> RuleViolation | None:
    minimum = rule["min_length"]
    maximum = rule["max_length"]

    if minimum <= len(content.title) <= maximum:
        return None

    return _make_violation(rule, field=rule["field"])


def _validate_uppercase_ratio(
    content: GeneratedContent,
    rule: Mapping[str, Any],
) -> RuleViolation | None:
    letters = [character for character in content.title if character.isalpha()]
    uppercase_ratio = (
        sum(character.isupper() for character in letters) / len(letters)
        if letters
        else 0.0
    )

    if uppercase_ratio <= rule["max_ratio"]:
        return None

    return _make_violation(rule, field=rule["field"])


def _validate_pattern(
    content: GeneratedContent,
    rule: Mapping[str, Any],
    *patterns: re.Pattern[str],
) -> RuleViolation | None:
    for field in rule["fields"]:
        value = getattr(content, field)
        if any(pattern.search(value) for pattern in patterns):
            return _make_violation(rule, field=field)

    return None


def _validate_emoji_spam(
    content: GeneratedContent,
    rule: Mapping[str, Any],
) -> RuleViolation | None:
    emoji_count = 0

    for field in rule["fields"]:
        value = getattr(content, field)
        emoji_count += len(EMOJI_PATTERN.findall(value))

        if emoji_count > rule["max_count"]:
            return _make_violation(rule, field=field)

    return None


def _make_violation(
    rule: Mapping[str, Any],
    field: str | None,
) -> RuleViolation:
    return RuleViolation(
        rule_id=rule["id"],
        message=rule["description"],
        severity=rule["severity"],
        field=field,
    )