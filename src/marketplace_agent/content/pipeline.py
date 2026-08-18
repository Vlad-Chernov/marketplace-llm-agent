from pathlib import Path
from typing import Any

import yaml

from marketplace_agent.catalog.extractor import extract_attributes
from marketplace_agent.content.generator import generate_content
from marketplace_agent.content.repair import repair_content
from marketplace_agent.data_generation.specs import load_attribute_specs
from marketplace_agent.domain.models import (
    GeneratedContent,
    PipelineResult,
    Product,
    RuleViolation,
)
from marketplace_agent.llm.base import LLMClient
from marketplace_agent.validation.deterministic import validate_deterministic
from marketplace_agent.validation.grounding import (
    GroundingEvidence,
    GroundingResult,
    validate_grounding,
)
from marketplace_agent.validation.semantic import validate_semantic


def run_content_pipeline(
    product: Product,
    llm: LLMClient,
    max_attempts: int = 3,
) -> PipelineResult:
    """Generate, validate and repair one product card."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")

    attribute_specs = load_attribute_specs(
        _project_root() / "data" / "specs" / f"{product.category}.yaml"
    )
    deterministic_rules, semantic_rules = _load_rules()
    extracted_attributes = extract_attributes(product, attribute_specs, llm)

    confirmed_attributes = {
        key: attribute.value
        for key, attribute in extracted_attributes.attributes.items()
        if attribute.value is not None
    }
    evidence = GroundingEvidence(
        confirmed_attributes=confirmed_attributes,
        supplier_description=product.supplier_description,
    )

    content: GeneratedContent | None = None
    violations: list[RuleViolation] = []
    seen_contents: set[str] = set()

    for attempt in range(1, max_attempts + 1):
        if content is None:
            content = generate_content(product, extracted_attributes, llm)
        else:
            content = repair_content(content, violations, evidence, llm)

        violations = _validate_content(
            content,
            deterministic_rules,
            semantic_rules,
            evidence,
            llm,
        )
        content_fingerprint = content.model_dump_json()

        if content_fingerprint in seen_contents:
            return PipelineResult(
                sku=product.sku,
                content=content,
                violations=violations,
                attempts=attempt,
                status="manual_review",
            )

        seen_contents.add(content_fingerprint)

        if not violations:
            return PipelineResult(
                sku=product.sku,
                content=content,
                attempts=attempt,
                status="completed",
            )

        if attempt == max_attempts:
            return PipelineResult(
                sku=product.sku,
                content=content,
                violations=violations,
                attempts=attempt,
                status="manual_review",
            )

    raise RuntimeError("Content pipeline ended without a result.")


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


def _load_rules() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rules_path = _project_root() / "data" / "processed" / "rules.yaml"
    rules = yaml.safe_load(rules_path.read_text(encoding="utf-8"))

    return (
        list(rules["deterministic_rules"]),
        list(rules["semantic_rules"]),
    )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]