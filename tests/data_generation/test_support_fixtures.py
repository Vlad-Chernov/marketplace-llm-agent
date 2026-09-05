from pathlib import Path

import yaml

RULES_PATH = Path("data/processed/rules.yaml")
SUPPORT_PATH = Path("data/support")


def test_rules_fixture_contains_deterministic_and_semantic_rules() -> None:
    data = yaml.safe_load(RULES_PATH.read_text(encoding="utf-8"))

    assert len(data["deterministic_rules"]) == 5
    assert "emoji-spam" in {
        rule["id"] for rule in data["deterministic_rules"]
}
    assert len(data["semantic_rules"]) == 15
    assert (
        len(data["deterministic_rules"])
        + len(data["semantic_rules"])
    ) == 20

    rule_ids = [
        rule["id"]
        for group in ("deterministic_rules", "semantic_rules")
        for rule in data[group]
    ]
    assert len(rule_ids) == len(set(rule_ids))


def test_support_fixture_contains_ten_documents() -> None:
    document_names = {path.name for path in SUPPORT_PATH.glob("*.md")}

    assert document_names == {
        "cancellation.md",
        "delivery.md",
        "exchange.md",
        "kettles.md",
        "laptops.md",
        "order_status.md",
        "payment.md",
        "returns.md",
        "sneakers.md",
        "warranty.md",
    }