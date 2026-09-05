import json
from pathlib import Path

import yaml

from marketplace_agent.domain.models import GeneratedContent
from marketplace_agent.evals.models import GoldenCase
from marketplace_agent.validation.deterministic import validate_deterministic

GOLD_CASES_PATH = Path("data/gold/mvp_cases.json")


def test_mvp_golden_dataset_has_required_coverage() -> None:
    raw_cases = json.loads(GOLD_CASES_PATH.read_text(encoding="utf-8"))
    cases = [GoldenCase.model_validate(raw_case) for raw_case in raw_cases]

    assert len(cases) == 50
    assert len({case.id for case in cases}) == 50
    assert all(case.input for case in cases)
    assert sum(case.origin == "manual" for case in cases) >= 30

    case_types = {case.type for case in cases}
    assert {
        "attribute_extraction",
        "validation",
        "content_generation",
        "review_analysis",
        "support",
        "adversarial",
    } <= case_types

    assert sum(
        "no_answer" in raw_case.get("tags", [])
        for raw_case in raw_cases
    ) >= 8


def test_attribute_cases_have_supplier_descriptions() -> None:
    raw_cases = json.loads(GOLD_CASES_PATH.read_text(encoding="utf-8"))
    cases = [GoldenCase.model_validate(raw_case) for raw_case in raw_cases]

    attribute_cases = [
        case for case in cases if case.type == "attribute_extraction"
    ]

    assert len(attribute_cases) == 14
    assert all(
        isinstance(case.input.get("supplier_description"), str)
        and case.input["supplier_description"]
        for case in attribute_cases
    )


def test_validation_cases_have_generated_content_inputs() -> None:
    raw_cases = json.loads(GOLD_CASES_PATH.read_text(encoding="utf-8"))
    cases = [GoldenCase.model_validate(raw_case) for raw_case in raw_cases]

    validation_cases = [
        case for case in cases if case.type == "validation"
    ]

    assert len(validation_cases) == 10
    assert all(
        {"title", "bullets", "description", "keywords", "used_attributes"}
        <= set(case.input.get("content", {}))
        for case in validation_cases
    )


def test_content_generation_cases_have_product_inputs() -> None:
    raw_cases = json.loads(GOLD_CASES_PATH.read_text(encoding="utf-8"))
    cases = [GoldenCase.model_validate(raw_case) for raw_case in raw_cases]

    content_cases = [
        case for case in cases if case.type == "content_generation"
    ]

    assert len(content_cases) == 4
    assert all(
        {
            "sku",
            "category",
            "brand",
            "model",
            "price",
            "sales_count",
            "supplier_description",
        } <= set(case.input.get("product", {}))
        for case in content_cases
    )


def test_review_cases_have_review_inputs() -> None:
    raw_cases = json.loads(GOLD_CASES_PATH.read_text(encoding="utf-8"))
    cases = [GoldenCase.model_validate(raw_case) for raw_case in raw_cases]

    review_cases = [
        case for case in cases if case.type == "review_analysis"
    ]

    assert len(review_cases) == 6
    assert all(
        {
            "review_id",
            "sku",
            "rating",
            "text",
            "created_at",
            "helpful_count",
        } <= set(case.input.get("review", {}))
        for case in review_cases
    )


def test_support_cases_have_message_and_session_inputs() -> None:
    raw_cases = json.loads(GOLD_CASES_PATH.read_text(encoding="utf-8"))
    cases = [GoldenCase.model_validate(raw_case) for raw_case in raw_cases]

    support_cases = [
        case
        for case in cases
        if case.type in {"support", "adversarial"}
    ]

    assert len(support_cases) == 16
    assert all(
        {"message", "session_id"} <= set(case.input)
        for case in support_cases
    )

def test_mvp_golden_manifest_matches_cases() -> None:
    manifest_path = Path("data/gold/MANIFEST.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_cases = json.loads(GOLD_CASES_PATH.read_text(encoding="utf-8"))

    assert manifest["suite"] == "mvp"
    assert manifest["split"] == "frozen"
    assert manifest["target_case_count"] == 50
    assert manifest["target_case_count"] == len(raw_cases)

def test_validation_cases_expect_deterministic_violations() -> None:
    raw_cases = json.loads(GOLD_CASES_PATH.read_text(encoding="utf-8"))
    rules_path = Path("data/processed/rules.yaml")
    rules = yaml.safe_load(rules_path.read_text(encoding="utf-8"))

    for raw_case in raw_cases:
        if raw_case["type"] != "validation":
            continue

        content = GeneratedContent.model_validate(raw_case["input"]["content"])
        actual_rule_ids = {
            violation.rule_id
            for violation in validate_deterministic(
                content,
                rules["deterministic_rules"],
            )
        }
        expected_rule_ids = set(
            raw_case["expected_answer"]["violations"]
        )

        assert actual_rule_ids <= expected_rule_ids