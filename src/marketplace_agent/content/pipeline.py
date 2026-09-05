from pathlib import Path
from typing import Any

import yaml

from marketplace_agent.catalog.extractor import extract_attributes
from marketplace_agent.content.graph import build_content_graph
from marketplace_agent.data_generation.specs import load_attribute_specs
from marketplace_agent.domain.models import (
    PipelineResult,
    Product,
)
from marketplace_agent.llm.base import LLMClient
from marketplace_agent.validation.grounding import GroundingEvidence


def run_content_pipeline(
    product: Product,
    llm: LLMClient,
    max_attempts: int = 3,
) -> PipelineResult:
    """Prepare one product and delegate its cycle to LangGraph."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")

    attribute_specs = load_attribute_specs(
        _project_root() / "data" / "specs" / f"{product.category}.yaml"
    )
    extracted_attributes = extract_attributes(
        product,
        attribute_specs,
        llm,
    )
    deterministic_rules, semantic_rules = _load_rules()
    confirmed_attributes = {
        key: attribute.value
        for key, attribute in extracted_attributes.attributes.items()
        if attribute.value is not None
    }
    evidence = GroundingEvidence(
        confirmed_attributes=confirmed_attributes,
        supplier_description=product.supplier_description,
    )

    graph = build_content_graph(llm)
    result = graph.invoke(
        {
            "product": product,
            "extracted_attributes": extracted_attributes,
            "evidence": evidence,
            "deterministic_rules": deterministic_rules,
            "semantic_rules": semantic_rules,
            "content": None,
            "violations": [],
            "attempts": 0,
            "max_attempts": max_attempts,
            "seen_contents": set(),
        }
    )

    return result["result"]


def _load_rules() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rules_path = _project_root() / "data" / "processed" / "rules.yaml"
    rules = yaml.safe_load(rules_path.read_text(encoding="utf-8"))

    return (
        list(rules["deterministic_rules"]),
        list(rules["semantic_rules"]),
    )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]