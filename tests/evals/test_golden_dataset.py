import json
from pathlib import Path

from marketplace_agent.evals.models import GoldenCase

GOLD_CASES_PATH = Path("data/gold/mvp_cases.json")


def test_mvp_golden_dataset_has_required_coverage() -> None:
    raw_cases = json.loads(GOLD_CASES_PATH.read_text(encoding="utf-8"))
    cases = [GoldenCase.model_validate(raw_case) for raw_case in raw_cases]

    assert len(cases) == 25
    assert len({case.id for case in cases}) == 25
    assert all(case.input for case in cases)
    assert sum(case.origin == "manual" for case in cases) >= 15

    case_types = {case.type for case in cases}
    assert {
        "attribute_extraction",
        "validation",
        "no_answer",
        "adversarial",
    } <= case_types

def test_attribute_cases_have_supplier_descriptions() -> None:
    raw_cases = json.loads(GOLD_CASES_PATH.read_text(encoding="utf-8"))
    cases = [GoldenCase.model_validate(raw_case) for raw_case in raw_cases]

    attribute_cases = [
        case for case in cases if case.type == "attribute_extraction"
    ]

    assert len(attribute_cases) == 9
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

    assert len(validation_cases) == 6
    assert all(
        {"title", "bullets", "description", "keywords", "used_attributes"}
        <= set(case.input.get("content", {}))
        for case in validation_cases
    )

def test_content_generation_case_has_product_input() -> None:
    raw_cases = json.loads(GOLD_CASES_PATH.read_text(encoding="utf-8"))
    cases = [GoldenCase.model_validate(raw_case) for raw_case in raw_cases]

    content_cases = [
        case for case in cases if case.type == "content_generation"
    ]

    assert len(content_cases) == 1
    assert {
        "sku",
        "category",
        "brand",
        "model",
        "price",
        "sales_count",
        "supplier_description",
    } <= set(content_cases[0].input.get("product", {}))

def test_review_case_has_review_input() -> None:
    raw_cases = json.loads(GOLD_CASES_PATH.read_text(encoding="utf-8"))
    cases = [GoldenCase.model_validate(raw_case) for raw_case in raw_cases]

    review_cases = [
        case for case in cases if case.type == "review_analysis"
    ]

    assert len(review_cases) == 1
    assert {
        "review_id",
        "sku",
        "rating",
        "text",
        "created_at",
        "helpful_count",
    } <= set(review_cases[0].input.get("review", {}))

def test_support_cases_have_message_and_session_inputs() -> None:
    raw_cases = json.loads(GOLD_CASES_PATH.read_text(encoding="utf-8"))
    cases = [GoldenCase.model_validate(raw_case) for raw_case in raw_cases]

    support_cases = [
        case
        for case in cases
        if case.type in {"support", "no_answer", "adversarial"}
    ]

    assert len(support_cases) == 8
    assert all(
        {"message", "session_id"} <= set(case.input)
        for case in support_cases
    )