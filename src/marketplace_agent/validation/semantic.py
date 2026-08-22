import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from marketplace_agent.domain.models import GeneratedContent, RuleViolation
from marketplace_agent.llm.base import LLMClient, Message
from marketplace_agent.llm.structured import chat_structured


class SemanticValidationError(ValueError):
    """Raised when an LLM violation cannot be linked to a known rule."""


class SemanticFinding(BaseModel):
    """One semantic violation detected by an LLM."""

    model_config = ConfigDict(extra="forbid")

    rule_id: str = Field(min_length=1)
    field: Literal["title", "description"]
    evidence: str = Field(min_length=1)


class SemanticValidationResponse(BaseModel):
    """Structured LLM response for semantic validation."""

    model_config = ConfigDict(extra="forbid")

    violations: list[SemanticFinding]


def validate_semantic(
    content: GeneratedContent,
    semantic_rules: Sequence[Mapping[str, Any]],
    llm: LLMClient,
) -> list[RuleViolation]:
    rules_by_id = {str(rule["id"]): rule for rule in semantic_rules}
    response = chat_structured(
        client=llm,
        messages=_build_messages(content, semantic_rules),
        response_schema=SemanticValidationResponse,
        max_retries=1,
    )

    violations: list[RuleViolation] = []

    for finding in response.violations:
        rule = rules_by_id.get(finding.rule_id)
        if rule is None:
            raise SemanticValidationError(
                f"LLM returned unknown rule: {finding.rule_id}"
            )

        violations.append(
            RuleViolation(
                rule_id=finding.rule_id,
                message=rule["description"],
                severity=rule["severity"],
                field=finding.field,
            )
        )

    return violations


def _build_messages(
    content: GeneratedContent,
    semantic_rules: Sequence[Mapping[str, Any]],
) -> list[Message]:
    rules_json = json.dumps(semantic_rules, ensure_ascii=False)
    content_json = json.dumps(
        {
            "title": content.title,
            "description": content.description,
        },
        ensure_ascii=False,
    )

    return [
        Message(
            role="system",
            content=(
                "Проверь карточку товара на смысловые нарушения. "
                "Верни только JSON по схеме: "
                '{"violations":[{"rule_id":"...","field":"title или description",'
                '"evidence":"точная проблемная фраза"}]}. '
                "Добавляй только нарушения из переданных правил. "
                f"Правила: {rules_json}"
            ),
        ),
        Message(role="user", content=f"Карточка: {content_json}"),
    ]