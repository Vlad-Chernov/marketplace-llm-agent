from pathlib import Path

import yaml

RULES_PATH = Path("data/processed/rules.yaml")
SUPPORT_PATH = Path("data/support")


def test_rules_fixture_contains_deterministic_and_semantic_rules() -> None:
    data = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))

    assert len(data["deterministic_rules"]) == 4
    assert len(data["semantic_rules"]) == 3

    rule_ids = [
        rule["id"]
        for group in ("deterministic_rules", "semantic_rules")
        for rule in data[group]
    ]
    assert len(rule_ids) == len(set(rule_ids))


def test_support_fixture_contains_four_documents() -> None:
    document_names = {path.name for path in SUPPORT_PATH.glob("*.md")}

    assert document_names == {
        "returns.md",
        "delivery.md",
        "warranty.md",
        "exchange.md",
    }