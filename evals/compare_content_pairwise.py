"""Prepare and evaluate blind legacy-vs-LangGraph content comparisons."""

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from random import Random
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marketplace_agent.config import Settings
from marketplace_agent.content.pipeline import run_content_pipeline
from marketplace_agent.data_generation.catalog import generate_clean_products
from marketplace_agent.data_generation.noise import noise_product
from marketplace_agent.domain.models import GeneratedContent
from marketplace_agent.evals.content_pairwise import (
    BlindContentPair,
    HumanPairChoice,
    evaluate_pairwise_agreement,
    judge_blind_pairs,
    parse_human_choices,
    serialize_blind_ballot,
    serialize_pairwise_result,
)
from marketplace_agent.evals.content_pipeline import run_pipeline_version
from marketplace_agent.evals.legacy_content_pipeline import (
    run_legacy_content_pipeline,
)
from marketplace_agent.llm.factory import create_llm_client


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare content cards with blind human and LLM judges."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--legacy-version", required=True)
    prepare.add_argument("--graph-version", required=True)
    prepare.add_argument("--catalog-seed", type=int, default=31)
    prepare.add_argument("--noise-seed", type=int, default=41)
    prepare.add_argument("--max-attempts", type=int, default=3)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--run-id", required=True)
    evaluate.add_argument("--choices", type=Path, required=True)
    return parser.parse_args()


def print_progress(event: dict[str, object]) -> None:
    print(
        f"[{event['event_type']}] {event.get('position')}/{event.get('total')} "
        f"{event.get('sku', event.get('pair_id'))}",
        flush=True,
    )


def main() -> None:
    arguments = parse_arguments()
    if arguments.command == "prepare":
        prepare(arguments)
    else:
        evaluate(arguments)


def prepare(arguments: argparse.Namespace) -> None:
    settings = Settings.from_environment()
    noise_rng = Random(arguments.noise_seed)
    products = [
        noise_product(product, noise_rng)
        for product in generate_clean_products(40, arguments.catalog_seed)
    ]
    progress = print_progress if os.getenv("LLM_PROGRESS") == "1" else None
    prices = {
        "input_price_per_million": settings.input_price_per_million,
        "output_price_per_million": settings.output_price_per_million,
    }
    legacy_results = run_pipeline_version(
        products, lambda: create_llm_client(settings), run_legacy_content_pipeline,
        arguments.max_attempts, **prices, version=arguments.legacy_version,
        progress=progress,
    )
    graph_results = run_pipeline_version(
        products, lambda: create_llm_client(settings), run_content_pipeline,
        arguments.max_attempts, **prices, version=arguments.graph_version,
        progress=progress,
    )
    run_id = uuid4().hex
    from marketplace_agent.evals.content_pairwise import build_blind_pairs

    pairs = build_blind_pairs(legacy_results, graph_results, run_id)
    manual_dir = PROJECT_ROOT / "evals/manual"
    manual_dir.mkdir(parents=True, exist_ok=True)
    ballot_path = manual_dir / f"content-pairs-{run_id}.json"
    ballot_path.write_text(
        json.dumps(serialize_blind_ballot(pairs), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    mapping_path = manual_dir / f"content-pairs-{run_id}-mapping.json"
    mapping_path.write_text(
        json.dumps(_serialize_mapping(pairs), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_run(
        run_id,
        {
            "created_at": datetime.now(UTC).isoformat(),
            "candidate_count": 40,
            "catalog_seed": arguments.catalog_seed,
            "noise_seed": arguments.noise_seed,
            "legacy_version": arguments.legacy_version,
            "graph_version": arguments.graph_version,
            "selected_pair_count": len(pairs),
        },
    )
    print(f"Ballot: {ballot_path}")
    print("Fill every choice with A, B, or tie before evaluate.")


def evaluate(arguments: argparse.Namespace) -> None:
    mapping_path = PROJECT_ROOT / "evals/manual" / (
        f"content-pairs-{arguments.run_id}-mapping.json"
    )
    pairs = _load_mapping(mapping_path)
    choices_payload = json.loads(arguments.choices.read_text(encoding="utf-8"))
    human_choices = parse_human_choices(choices_payload, pairs)
    settings = Settings.from_environment()
    progress = print_progress if os.getenv("LLM_PROGRESS") == "1" else None
    judged_pairs = judge_blind_pairs(
        pairs, lambda: create_llm_client(settings), progress
    )
    agreement = evaluate_pairwise_agreement(human_choices, judged_pairs)
    result = serialize_pairwise_result(pairs, human_choices, judged_pairs)
    result["agreement"] = agreement.__dict__
    result["human_version_wins"] = _version_wins(human_choices, pairs)
    result["judge_version_wins"] = _version_wins(
        [
            HumanPairChoice(item.pair_id, item.choice)
            for item in judged_pairs
            if item.choice is not None
        ],
        pairs,
    )
    _write_run(arguments.run_id, {"result": result})
    print(f"Comparable pairs: {agreement.comparable_pair_count}")
    print(f"Exact agreement: {agreement.exact_agreement:.3f}")
    print(f"Cohen's kappa: {agreement.cohens_kappa:.3f}")


def _serialize_mapping(pairs: list[BlindContentPair]) -> dict[str, object]:
    return {
        "pairs": [
            {
                "pair_id": pair.pair_id,
                "sku": pair.sku,
                "confirmed_attributes": dict(pair.confirmed_attributes),
                "card_a": pair.card_a.model_dump(),
                "card_b": pair.card_b.model_dump(),
                "a_version": pair.a_version,
                "b_version": pair.b_version,
            }
            for pair in pairs
        ]
    }


def _load_mapping(path: Path) -> list[BlindContentPair]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        BlindContentPair(
            pair_id=record["pair_id"],
            sku=record["sku"],
            confirmed_attributes=record["confirmed_attributes"],
            card_a=GeneratedContent.model_validate(record["card_a"]),
            card_b=GeneratedContent.model_validate(record["card_b"]),
            a_version=record["a_version"],
            b_version=record["b_version"],
        )
        for record in payload["pairs"]
    ]


def _version_wins(
    choices: list[HumanPairChoice], pairs: list[BlindContentPair]
) -> dict[str, int]:
    pairs_by_id = {pair.pair_id: pair for pair in pairs}
    wins = {"legacy": 0, "langgraph": 0, "tie": 0}
    for choice in choices:
        if choice.choice == "tie":
            wins["tie"] += 1
        elif choice.choice == "A":
            wins[pairs_by_id[choice.pair_id].a_version] += 1
        else:
            wins[pairs_by_id[choice.pair_id].b_version] += 1
    return wins


def _write_run(run_id: str, payload: dict[str, object]) -> None:
    path = PROJECT_ROOT / "evals/runs" / f"content-pairwise-{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    path.write_text(
        json.dumps(existing | payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
