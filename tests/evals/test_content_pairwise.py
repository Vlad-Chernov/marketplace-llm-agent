import pytest

from marketplace_agent.domain.models import GeneratedContent
from marketplace_agent.evals.content_pairwise import (
    build_blind_pairs,
    parse_human_choices,
    serialize_blind_ballot,
)
from marketplace_agent.evals.content_pipeline import (
    ContentPipelineCaseResult,
)


def make_content(version: str, sku: str) -> GeneratedContent:
    return GeneratedContent(
        title=f"{version} {sku}",
        bullets=[f"Пункт {version}"],
        description=f"Описание {version}.",
        keywords=["ноутбук"],
        used_attributes={"ram_gb": "16"},
    )


def make_results(version: str) -> list[ContentPipelineCaseResult]:
    return [
        ContentPipelineCaseResult(
            sku=f"LAP-000{number}",
            true_attributes={"ram_gb": "16"},
            extracted_attributes={},
            used_attributes={"ram_gb": "16"},
            violations=[],
            latency_ms=0,
            cost_usd=0.0,
            content=make_content(version, f"LAP-000{number}"),
        )
        for number in range(1, 4)
    ]


def test_builds_reproducible_blind_pairs_without_version_fields() -> None:
    pairs = build_blind_pairs(
        make_results("legacy"),
        make_results("graph"),
        run_id="fixed-run",
        required_pair_count=2,
    )

    ballot = serialize_blind_ballot(pairs)

    assert [pair.pair_id for pair in pairs] == ["PAIR-001", "PAIR-002"]
    assert ballot["pairs"][0]["choice"] is None
    assert "a_version" not in ballot["pairs"][0]
    assert "latency_ms" not in ballot["pairs"][0]
    assert "trace" not in ballot["pairs"][0]


def test_rejects_invalid_or_incomplete_human_choices() -> None:
    pairs = build_blind_pairs(
        make_results("legacy"),
        make_results("graph"),
        run_id="fixed-run",
        required_pair_count=2,
    )

    with pytest.raises(ValueError, match="exactly one choice"):
        parse_human_choices(
            {"choices": [{"pair_id": "PAIR-001", "choice": "A"}]},
            pairs,
        )

    with pytest.raises(ValueError, match="A, B, or tie"):
        parse_human_choices(
            {
                "choices": [
                    {"pair_id": "PAIR-001", "choice": "legacy"},
                    {"pair_id": "PAIR-002", "choice": "B"},
                ]
            },
            pairs,
        )


def test_rejects_when_not_enough_completed_pairs() -> None:
    graph_results = make_results("graph")
    graph_results[1] = ContentPipelineCaseResult(
        sku="LAP-0002",
        true_attributes={"ram_gb": "16"},
        extracted_attributes={},
        used_attributes={},
        violations=[],
        latency_ms=0,
        cost_usd=0.0,
        status="manual_review",
    )

    with pytest.raises(ValueError, match="Fewer than 3 completed"):
        build_blind_pairs(
            make_results("legacy"),
            graph_results,
            run_id="fixed-run",
            required_pair_count=3,
        )
