"""Blind pair construction for content-pipeline evaluation."""

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

from marketplace_agent.domain.models import GeneratedContent
from marketplace_agent.evals.content_pipeline import (
    ContentPipelineCaseResult,
)

Choice = Literal["A", "B", "tie"]


@dataclass(frozen=True)
class BlindContentPair:
    """One blind comparison between legacy and graph content."""

    pair_id: str
    sku: str
    confirmed_attributes: Mapping[str, Any]
    card_a: GeneratedContent
    card_b: GeneratedContent
    a_version: str
    b_version: str


@dataclass(frozen=True)
class HumanPairChoice:
    """One validated manual choice for a blind pair."""

    pair_id: str
    choice: Choice


def build_blind_pairs(
    legacy_results: Sequence[ContentPipelineCaseResult],
    graph_results: Sequence[ContentPipelineCaseResult],
    run_id: str,
    required_pair_count: int = 30,
) -> list[BlindContentPair]:
    """Select completed pairs and deterministically assign A/B positions."""

    legacy_skus = [result.sku for result in legacy_results]
    graph_skus = [result.sku for result in graph_results]
    if legacy_skus != graph_skus:
        raise ValueError("Legacy and graph results must use the same SKUs.")

    pairs: list[BlindContentPair] = []
    for legacy, graph in zip(legacy_results, graph_results, strict=True):
        if not _has_completed_content(legacy) or not _has_completed_content(graph):
            continue

        pair_id = f"PAIR-{len(pairs) + 1:03d}"
        if _legacy_is_card_a(run_id, pair_id):
            card_a, card_b = legacy.content, graph.content
            a_version, b_version = "legacy", "langgraph"
        else:
            card_a, card_b = graph.content, legacy.content
            a_version, b_version = "langgraph", "legacy"

        pairs.append(
            BlindContentPair(
                pair_id=pair_id,
                sku=legacy.sku,
                confirmed_attributes=legacy.true_attributes,
                card_a=card_a,
                card_b=card_b,
                a_version=a_version,
                b_version=b_version,
            )
        )
        if len(pairs) == required_pair_count:
            return pairs

    raise ValueError(
        f"Fewer than {required_pair_count} completed content pairs are available."
    )


def serialize_blind_ballot(
    pairs: Sequence[BlindContentPair],
) -> dict[str, object]:
    """Serialize only information a human evaluator may see."""

    return {
        "pairs": [
            {
                "pair_id": pair.pair_id,
                "sku": pair.sku,
                "confirmed_attributes": dict(pair.confirmed_attributes),
                "card_a": _serialize_visible_card(pair.card_a),
                "card_b": _serialize_visible_card(pair.card_b),
                "choice": None,
            }
            for pair in pairs
        ]
    }


def parse_human_choices(
    payload: Mapping[str, object],
    pairs: Sequence[BlindContentPair],
) -> list[HumanPairChoice]:
    """Validate one manual A/B/tie choice for every pair."""

    raw_choices = payload.get("choices")
    if not isinstance(raw_choices, list):
        raise TypeError("Choices must be a list.")

    expected_ids = {pair.pair_id for pair in pairs}
    parsed_choices: list[HumanPairChoice] = []
    seen_ids: set[str] = set()

    for raw_choice in raw_choices:
        if not isinstance(raw_choice, Mapping):
            raise TypeError("Each choice must be an object.")
        pair_id = raw_choice.get("pair_id")
        choice = raw_choice.get("choice")
        if not isinstance(pair_id, str) or pair_id in seen_ids:
            raise ValueError("Every pair must have exactly one choice.")
        if choice not in {"A", "B", "tie"}:
            raise ValueError("Choice must be A, B, or tie.")
        seen_ids.add(pair_id)
        parsed_choices.append(
            HumanPairChoice(pair_id=pair_id, choice=cast("Choice", choice))
        )

    if seen_ids != expected_ids:
        raise ValueError("Every pair must have exactly one choice.")
    return parsed_choices


def _has_completed_content(result: ContentPipelineCaseResult) -> bool:
    return result.status == "completed" and result.content is not None


def _legacy_is_card_a(run_id: str, pair_id: str) -> bool:
    digest = hashlib.sha256(f"{run_id}:{pair_id}".encode()).digest()
    return digest[0] % 2 == 0


def _serialize_visible_card(content: GeneratedContent) -> dict[str, object]:
    return {
        "title": content.title,
        "bullets": content.bullets,
        "description": content.description,
        "keywords": content.keywords,
    }
