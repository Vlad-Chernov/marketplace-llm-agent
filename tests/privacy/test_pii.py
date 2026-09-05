import json
from pathlib import Path

from marketplace_agent.privacy.pii import PiiRedactor

PII_CASES_PATH = Path("data/gold/pii_cases.json")


def test_detects_and_redacts_all_manual_pii_cases() -> None:
    cases = json.loads(PII_CASES_PATH.read_text(encoding="utf-8"))

    assert len(cases) == 150
    assert all(case["origin"] == "manual" for case in cases)

    redactor = PiiRedactor()
    with redactor:
        for case in cases:
            expected_types = set(case["expected_types"])

            assert redactor.detect_types(case["text"]) == expected_types

            redacted = redactor.redact(case["text"])
            assert redacted != case["text"]

            for pii_type in expected_types:
                assert f"[{pii_type.upper()}]" in redacted

    assert redactor._nlp is None