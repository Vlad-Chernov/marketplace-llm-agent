import json
from pathlib import Path

from marketplace_agent.evals.models import GoldenCase

ADVERSARIAL_CASES_PATH = Path(
    "data/gold/adversarial_cases.json"
)


def test_adversarial_dataset_has_full_manual_coverage() -> None:
    raw_cases = json.loads(
        ADVERSARIAL_CASES_PATH.read_text(encoding="utf-8")
    )
    cases = [
        GoldenCase.model_validate(raw_case)
        for raw_case in raw_cases
    ]

    assert len(cases) == 30
    assert len({case.id for case in cases}) == 30
    assert all(case.type == "adversarial" for case in cases)
    assert all(case.origin == "manual" for case in cases)
    assert all(
        case.expected_answer == {"status": "escalated"}
        for case in cases
    )
    assert all(case.must_call_tools == [] for case in cases)
    assert {
        "direct_injection",
        "indirect_injection",
        "cross_order",
        "privacy_bypass",
        "forbidden_action",
    } == {
        tag
        for case in cases
        for tag in case.tags
    }