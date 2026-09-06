from marketplace_agent.content.examples import (
    ContentExample,
    select_similar_examples,
)
from marketplace_agent.domain.models import GeneratedContent


def make_example(
    example_id: str,
    attributes: dict[str, str],
) -> ContentExample:
    return ContentExample(
        example_id=example_id,
        category="laptop",
        confirmed_attributes=attributes,
        content=GeneratedContent(
            title="Ноутбук для работы",
            bullets=["Надёжная конфигурация"],
            description="Карточка-пример.",
            keywords=["ноутбук"],
            used_attributes=attributes,
        ),
    )


def test_selects_examples_with_most_matching_attributes() -> None:
    selected = select_similar_examples(
        category="laptop",
        confirmed_attributes={
            "ram_gb": "16",
            "storage_gb": "512",
        },
        examples=[
            make_example("EXAMPLE-003", {"ram_gb": "8"}),
            make_example("EXAMPLE-002", {"ram_gb": "16"}),
            make_example(
                "EXAMPLE-001",
                {
                    "ram_gb": "16",
                    "storage_gb": "512",
                },
            ),
        ],
    )

    assert [example.example_id for example in selected] == [
        "EXAMPLE-001",
        "EXAMPLE-002",
    ]


def test_resolves_equal_scores_by_example_id() -> None:
    selected = select_similar_examples(
        category="laptop",
        confirmed_attributes={"ram_gb": "16"},
        examples=[
            make_example("EXAMPLE-002", {"ram_gb": "16"}),
            make_example("EXAMPLE-001", {"ram_gb": "16"}),
        ],
    )

    assert [example.example_id for example in selected] == [
        "EXAMPLE-001",
        "EXAMPLE-002",
    ]

from pathlib import Path

from marketplace_agent.content.examples import load_content_examples


def test_loads_eight_gold_content_examples() -> None:
    examples = load_content_examples(
        Path("data/gold/content_examples.json")
    )

    assert len(examples) == 8
    assert [example.example_id for example in examples] == [
        "EXAMPLE-001",
        "EXAMPLE-002",
        "EXAMPLE-003",
        "EXAMPLE-004",
        "EXAMPLE-005",
        "EXAMPLE-006",
        "EXAMPLE-007",
        "EXAMPLE-008",
    ]

import pytest


def test_rejects_example_with_unconfirmed_used_attribute() -> None:
    with pytest.raises(
        ValueError,
        match="used_attributes must contain only confirmed values",
    ):
        ContentExample(
            example_id="EXAMPLE-INVALID",
            category="laptop",
            confirmed_attributes={"ram_gb": "16"},
            content=GeneratedContent(
                title="Ноутбук",
                bullets=["16 GB RAM"],
                description="Карточка-пример.",
                keywords=["ноутбук"],
                used_attributes={
                    "ram_gb": "16",
                    "storage_gb": "512",
                },
            ),
        )