import json
from pathlib import Path

from marketplace_agent.domain.models import GeneratedContent, RuleViolation
from marketplace_agent.llm.base import LLMClient, Message
from marketplace_agent.llm.structured import chat_structured
from marketplace_agent.validation.grounding import GroundingEvidence


def repair_content(
    content: GeneratedContent,
    violations: list[RuleViolation],
    evidence: GroundingEvidence,
    llm: LLMClient,
) -> GeneratedContent:
    """Repair a generated card using explicit validation violations."""

    prompt_path = Path(__file__).parent / "prompts" / "repair_content.md"
    prompt = prompt_path.read_text(encoding="utf-8")

    prompt = prompt.replace(
        "{content}",
        json.dumps(content.model_dump(), ensure_ascii=False),
    )
    prompt = prompt.replace(
        "{violations}",
        json.dumps(
            [violation.model_dump() for violation in violations],
            ensure_ascii=False,
        ),
    )
    prompt = prompt.replace(
        "{evidence}",
        json.dumps(evidence.model_dump(), ensure_ascii=False),
    )

    return chat_structured(
        client=llm,
        messages=[Message(role="user", content=prompt)],
        response_schema=GeneratedContent,
        max_retries=1,
    )