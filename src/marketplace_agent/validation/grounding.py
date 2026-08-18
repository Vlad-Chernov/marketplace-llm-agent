import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from marketplace_agent.domain.models import GeneratedContent
from marketplace_agent.llm.base import LLMClient, Message
from marketplace_agent.llm.structured import chat_structured


class GroundingEvidence(BaseModel):
    """Evidence permitted for validating generated product content."""

    model_config = ConfigDict(extra="forbid")

    confirmed_attributes: dict[str, Any]
    supplier_description: str = Field(min_length=1)


class UnsupportedClaim(BaseModel):
    """One product-card claim that has no permitted evidence."""

    model_config = ConfigDict(extra="forbid")

    field: Literal[
        "title", "bullets", "description", "keywords", "used_attributes"
    ]
    claim: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class GroundingResult(BaseModel):
    """Unsupported claims found in a generated product card."""

    model_config = ConfigDict(extra="forbid")

    unsupported_claims: list[UnsupportedClaim]


def validate_grounding(
    content: GeneratedContent,
    evidence: GroundingEvidence,
    llm: LLMClient,
) -> GroundingResult:
    """Return product-card claims unsupported by permitted evidence."""

    unsupported_claims = _validate_used_attributes(
        content.used_attributes,
        evidence.confirmed_attributes,
    )
    if unsupported_claims:
        return GroundingResult(unsupported_claims=unsupported_claims)

    return chat_structured(
        client=llm,
        messages=_build_messages(content, evidence),
        response_schema=GroundingResult,
        max_retries=0,
    )


def _validate_used_attributes(
    used_attributes: dict[str, Any],
    confirmed_attributes: dict[str, Any],
) -> list[UnsupportedClaim]:
    unsupported_claims: list[UnsupportedClaim] = []

    for key, value in used_attributes.items():
        if key not in confirmed_attributes:
            reason = "Атрибут отсутствует в подтверждённых данных."
        elif value != confirmed_attributes[key]:
            reason = (
                "Значение не совпадает с подтверждённым: "
                f"{confirmed_attributes[key]!r}."
            )
        else:
            continue

        unsupported_claims.append(
            UnsupportedClaim(
                field="used_attributes",
                claim=f"{key}={value!r}",
                reason=reason,
            )
        )

    return unsupported_claims

def _build_messages(
    content: GeneratedContent,
    evidence: GroundingEvidence,
) -> list[Message]:
    """Build the LLM request from allowed evidence and card text."""

    evidence_json = json.dumps(
        {
            "confirmed_attributes": evidence.confirmed_attributes,
            "supplier_description": evidence.supplier_description,
        },
        ensure_ascii=False,
    )
    content_json = json.dumps(
        {
            "title": content.title,
            "bullets": content.bullets,
            "description": content.description,
            "keywords": content.keywords,
        },
        ensure_ascii=False,
    )

    return [
        Message(
            role="system",
            content=(
                "Проверь, подтверждён ли каждый атомарный факт карточки "
                "только переданными доказательствами. Не проверяй "
                "used_attributes: они уже проверены кодом. Верни только JSON "
                'по схеме: {"unsupported_claims":[{"field":"title, bullets, '
                'description или keywords","claim":"...","reason":"..."}]}. '
                'Если неподтверждённых фактов нет, верни '
                '{"unsupported_claims": []}. '
                f"Доказательства: {evidence_json}"
            ),
        ),
        Message(role="user", content=f"Карточка: {content_json}"),
    ]