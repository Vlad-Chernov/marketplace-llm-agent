import json
from pathlib import Path

from marketplace_agent.evals.models import GoldenCase

GOLD_CASES_PATH = Path("data/gold/mvp_cases.json")


def test_mvp_golden_dataset_has_required_coverage() -> None:
    raw_cases = json.loads(GOLD_CASES_PATH.read_text(encoding="utf-8"))
    cases = [GoldenCase.model_validate(raw_case) for raw_case in raw_cases]

    assert len(cases) == 25
    assert len({case.id for case in cases}) == 25
    assert sum(case.origin == "manual" for case in cases) >= 15

    case_types = {case.type for case in cases}
    assert {
        "attribute_extraction",
        "validation",
        "no_answer",
        "adversarial",
    } <= case_types