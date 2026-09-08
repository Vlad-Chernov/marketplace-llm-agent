from collections.abc import Sequence
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from marketplace_agent.catalog.extractor import (
    AttributeExtractionResult,
)
from marketplace_agent.content.examples import (
    ContentExample,
    select_similar_examples,
)
from marketplace_agent.content.generator import generate_content
from marketplace_agent.content.repair import repair_content
from marketplace_agent.domain.models import (
    GeneratedContent,
    PipelineAttempt,
    PipelineResult,
    Product,
    RuleViolation,
)
from marketplace_agent.llm.base import LLMClient
from marketplace_agent.validation.deterministic import (
    validate_deterministic,
)
from marketplace_agent.validation.grounding import (
    GroundingEvidence,
    GroundingResult,
    validate_grounding,
)
from marketplace_agent.validation.semantic import validate_semantic


class ContentGraphState(TypedDict, total=False):
    selected_example_ids: list[str]
    product: Product
    extracted_attributes: AttributeExtractionResult
    evidence: GroundingEvidence
    deterministic_rules: list[dict[str, Any]]
    semantic_rules: list[dict[str, Any]]
    content: GeneratedContent | None
    violations: list[RuleViolation]
    attempts: int
    attempt_history: list[PipelineAttempt]
    max_attempts: int
    seen_contents: set[str]
    result: PipelineResult


def build_content_graph(
    llm: LLMClient,
    *,
    examples: Sequence[ContentExample] = (),
):
    """Build the generate-validate-repair graph."""

    graph = StateGraph(ContentGraphState)
    graph.add_node("generate", _generate(llm, examples))
    graph.add_node("validate", _validate(llm))
    graph.add_node("remember", _remember)
    graph.add_node("repair", _repair(llm))
    graph.add_node("completed", _completed)
    graph.add_node("manual_review", _manual_review)

    graph.add_edge(START, "generate")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges(
        "validate",
        _next_step,
        {
            "completed": "completed",
            "manual_review": "manual_review",
            "remember": "remember",
        },
    )
    graph.add_edge("remember", "repair")
    graph.add_edge("repair", "validate")
    graph.add_edge("completed", END)
    graph.add_edge("manual_review", END)

    return graph.compile()


def _generate(
    llm: LLMClient,
    examples: Sequence[ContentExample],
):
    def generate(
        state: ContentGraphState,
    ) -> dict[str, object]:
        confirmed_attributes = {
            key: attribute.value
            for key, attribute in state[
                "extracted_attributes"
            ].attributes.items()
            if attribute.value is not None
        }
        selected_examples = select_similar_examples(
            category=state["product"].category,
            confirmed_attributes=confirmed_attributes,
            examples=examples,
        )

        return {
            "content": generate_content(
                state["product"],
                state["extracted_attributes"],
                llm,
                examples=selected_examples,
            ),
            "attempts": state["attempts"] + 1,
            "selected_example_ids": [
                example.example_id
                for example in selected_examples
            ],
        }

    return generate


def _validate(llm: LLMClient):
    def validate(
        state: ContentGraphState,
    ) -> dict[str, list[RuleViolation]]:
        content = state["content"]
        assert content is not None

        violations = _validate_content(
            content,
            state["deterministic_rules"],
            state["semantic_rules"],
            state["evidence"],
            llm,
        )
        return {
            "violations": violations,
            "attempt_history": [
                *state.get("attempt_history", []),
                PipelineAttempt(
                    attempt=state["attempts"],
                    violation_count=len(violations),
                    status="invalid" if violations else "valid",
                ),
            ],
        }

    return validate


def _next_step(
    state: ContentGraphState,
) -> Literal["completed", "manual_review", "remember"]:
    if not state["violations"]:
        return "completed"

    content = state["content"]
    assert content is not None
    fingerprint = content.model_dump_json()

    if fingerprint in state["seen_contents"]:
        return "manual_review"

    if state["attempts"] >= state["max_attempts"]:
        return "manual_review"

    return "remember"


def _remember(
    state: ContentGraphState,
) -> dict[str, set[str]]:
    content = state["content"]
    assert content is not None

    return {
        "seen_contents": {
            *state["seen_contents"],
            content.model_dump_json(),
        }
    }


def _repair(llm: LLMClient):
    def repair(
        state: ContentGraphState,
    ) -> dict[str, object]:
        content = state["content"]
        assert content is not None

        return {
            "content": repair_content(
                content,
                state["violations"],
                state["evidence"],
                llm,
            ),
            "attempts": state["attempts"] + 1,
        }

    return repair


def _completed(
    state: ContentGraphState,
) -> dict[str, PipelineResult]:
    content = state["content"]
    assert content is not None

    return {
        "result": PipelineResult(
            sku=state["product"].sku,
            content=content,
            attempts=state["attempts"],
            status="completed",
            selected_example_ids=state["selected_example_ids"],
            attempt_history=state.get("attempt_history", []),
        )
    }


def _manual_review(
    state: ContentGraphState,
) -> dict[str, PipelineResult]:
    content = state["content"]
    assert content is not None

    return {
        "result": PipelineResult(
            sku=state["product"].sku,
            content=content,
            violations=state["violations"],
            attempts=state["attempts"],
            status="manual_review",
            selected_example_ids=state["selected_example_ids"],
            attempt_history=state.get("attempt_history", []),
        )
    }


def _validate_content(
    content: GeneratedContent,
    deterministic_rules: list[dict[str, Any]],
    semantic_rules: list[dict[str, Any]],
    evidence: GroundingEvidence,
    llm: LLMClient,
) -> list[RuleViolation]:
    violations = validate_deterministic(content, deterministic_rules)

    grounding_result = validate_grounding(content, evidence, llm)
    violations.extend(_grounding_violations(grounding_result))

    violations.extend(validate_semantic(content, semantic_rules, llm))
    return violations


def _grounding_violations(
    grounding_result: GroundingResult,
) -> list[RuleViolation]:
    return [
        RuleViolation(
            rule_id="unsupported-claim",
            message=f"{claim.claim}: {claim.reason}",
            severity="high",
            field=claim.field,
        )
        for claim in grounding_result.unsupported_claims
    ]
