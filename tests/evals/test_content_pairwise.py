import json
from argparse import Namespace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import pytest

from marketplace_agent.domain.models import GeneratedContent
from marketplace_agent.evals.content_pairwise import (
    HumanPairChoice,
    JudgedPair,
    build_blind_pairs,
    evaluate_pairwise_agreement,
    judge_blind_pairs,
    parse_human_choices,
    serialize_blind_ballot,
    serialize_pairwise_result,
    serialize_probe_result,
    summarize_pair_availability,
)
from marketplace_agent.evals.content_pipeline import (
    ContentPipelineCaseResult,
)
from marketplace_agent.llm.base import FakeLLMClient, LLMResponse, Message


def load_comparison_cli():
    path = Path(__file__).parents[2] / "evals/compare_content_pairwise.py"
    spec = spec_from_file_location("compare_content_pairwise", path)
    assert spec is not None
    assert spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecordingFakeLLMClient(FakeLLMClient):
    def __init__(self, chat_responses: list[LLMResponse]) -> None:
        super().__init__(chat_responses=chat_responses)
        self.last_messages: list[Message] = []

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, object]] | None,
        response_schema: type[object] | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        self.last_messages = messages
        return super().chat(
            messages,
            tools,
            response_schema,
            temperature,
            max_tokens,
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


def test_parses_choices_from_completed_blind_ballot() -> None:
    pairs = build_blind_pairs(
        make_results("legacy"),
        make_results("graph"),
        run_id="fixed-run",
        required_pair_count=2,
    )
    ballot = serialize_blind_ballot(pairs)
    ballot_pairs = ballot["pairs"]
    assert isinstance(ballot_pairs, list)
    assert isinstance(ballot_pairs[0], dict)
    assert isinstance(ballot_pairs[1], dict)
    ballot_pairs[0]["choice"] = "A"
    ballot_pairs[1]["choice"] = "tie"

    choices = parse_human_choices(ballot, pairs)

    assert choices == [
        HumanPairChoice(pair_id="PAIR-001", choice="A"),
        HumanPairChoice(pair_id="PAIR-002", choice="tie"),
    ]

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


def test_judges_each_pair_with_new_client_and_blind_prompt() -> None:
    pairs = build_blind_pairs(
        make_results("legacy"),
        make_results("graph"),
        run_id="fixed-run",
        required_pair_count=2,
    )
    clients: list[RecordingFakeLLMClient] = []

    def factory() -> RecordingFakeLLMClient:
        client = RecordingFakeLLMClient(
            [
                LLMResponse(
                    content='{"choice":"A","reason":"Точнее."}',
                    model="fake-model",
                    prompt_tokens=1,
                    completion_tokens=1,
                )
            ]
        )
        clients.append(client)
        return client

    decisions = judge_blind_pairs(pairs, factory)

    assert len(clients) == 2
    assert [item.choice for item in decisions] == ["A", "A"]
    assert "a_version" not in clients[0].last_messages[0].content
    assert "b_version" not in clients[0].last_messages[0].content


def test_calculates_exact_agreement_and_kappa() -> None:
    metrics = evaluate_pairwise_agreement(
        [
            HumanPairChoice("PAIR-001", "A"),
            HumanPairChoice("PAIR-002", "A"),
            HumanPairChoice("PAIR-003", "B"),
            HumanPairChoice("PAIR-004", "B"),
        ],
        [
            JudgedPair("PAIR-001", "A"),
            JudgedPair("PAIR-002", "B"),
            JudgedPair("PAIR-003", "A"),
            JudgedPair("PAIR-004", "B"),
        ],
    )

    assert metrics.comparable_pair_count == 4
    assert metrics.exact_agreement == pytest.approx(0.5)
    assert metrics.cohens_kappa == pytest.approx(0.0)


def test_serializes_result_without_hidden_mapping_or_reasons() -> None:
    pairs = build_blind_pairs(
        make_results("legacy"),
        make_results("graph"),
        run_id="fixed-run",
        required_pair_count=2,
    )

    payload = serialize_pairwise_result(
        pairs=pairs,
        human_choices=[
            HumanPairChoice("PAIR-001", "A"),
            HumanPairChoice("PAIR-002", "tie"),
        ],
        judged_pairs=[
            JudgedPair("PAIR-001", "A"),
            JudgedPair("PAIR-002", None, error_type="RuntimeError"),
        ],
    )

    assert payload["human_choices"] == {"A": 1, "B": 0, "tie": 1}
    assert payload["judge_error_types"] == {"RuntimeError": 1}
    assert "a_version" not in str(payload)
    assert "b_version" not in str(payload)
    assert "reason" not in str(payload)


def test_summarizes_errors_when_pairs_are_unavailable() -> None:
    graph_results = make_results("graph")
    graph_results[0] = ContentPipelineCaseResult(
        sku="LAP-0001",
        true_attributes={"ram_gb": "16"},
        extracted_attributes={},
        used_attributes={},
        violations=[],
        latency_ms=0,
        cost_usd=0.0,
        status="error",
        error="LLMProviderError: HTTP 429",
    )

    summary = summarize_pair_availability(
        make_results("legacy"),
        graph_results,
    )

    assert summary == {
        "completed_pair_count": 2,
        "legacy_failures": {},
        "graph_failures": {"LLMProviderError": 1},
    }


def test_serializes_probe_error_for_one_pipeline_case() -> None:
    result = ContentPipelineCaseResult(
        sku="LAP-0001",
        true_attributes={},
        extracted_attributes={},
        used_attributes={},
        violations=[],
        latency_ms=123,
        cost_usd=0.0,
        status="error",
        error="LLMProviderError: HTTP 402",
    )

    assert serialize_probe_result(result) == {
        "sku": "LAP-0001",
        "status": "error",
        "error": "LLMProviderError: HTTP 402",
        "latency_ms": 123,
    }


def test_prepare_uses_fifty_candidates_and_selects_thirty_pairs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    compare_content_pairwise = load_comparison_cli()
    generated_counts: list[tuple[int, int]] = []

    def generate_products(count: int, seed: int) -> list[object]:
        generated_counts.append((count, seed))
        return [object() for _ in range(count)]

    def run_version(products: list[object], *_args, **_kwargs):
        return [
            ContentPipelineCaseResult(
                sku=f"LAP-{number:04d}",
                true_attributes={"ram_gb": "16"},
                extracted_attributes={},
                used_attributes={"ram_gb": "16"},
                violations=[],
                latency_ms=0,
                cost_usd=0.0,
                content=make_content("card", f"LAP-{number:04d}"),
            )
            for number in range(1, len(products) + 1)
        ]

    monkeypatch.setattr(
        compare_content_pairwise,
        "PROJECT_ROOT",
        tmp_path,
    )
    monkeypatch.setattr(
        compare_content_pairwise.Settings,
        "from_environment",
        staticmethod(
            lambda: SimpleNamespace(
                input_price_per_million=0.0,
                output_price_per_million=0.0,
            )
        ),
    )
    monkeypatch.setattr(
        compare_content_pairwise,
        "generate_clean_products",
        generate_products,
    )
    monkeypatch.setattr(
        compare_content_pairwise,
        "noise_product",
        lambda product, _: product,
    )
    monkeypatch.setattr(
        compare_content_pairwise,
        "run_pipeline_version",
        run_version,
    )

    compare_content_pairwise.prepare(
        Namespace(
            catalog_seed=31,
            noise_seed=41,
            max_attempts=3,
            legacy_version="legacy-linear-v1",
            graph_version="langgraph-v1",
        )
    )

    assert generated_counts == [(50, 31)]
    payload = next((tmp_path / "evals/runs").glob("*.json")).read_text()
    assert '"candidate_count": 50' in payload
    ballot = next((tmp_path / "evals/manual").glob("*.json"))
    assert len(json.loads(ballot.read_text())["pairs"]) == 30
