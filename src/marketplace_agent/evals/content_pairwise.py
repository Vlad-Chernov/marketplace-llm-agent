"""Blind pair construction for content-pipeline evaluation."""

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from marketplace_agent.domain.models import GeneratedContent
from marketplace_agent.evals.content_pipeline import (
    ContentPipelineCaseResult,
)
from marketplace_agent.llm.base import LLMClient, Message
from marketplace_agent.llm.structured import chat_structured

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


class PairJudgeDecision(BaseModel):
    """Structured blind decision returned by the LLM judge."""

    model_config = ConfigDict(extra="forbid")

    choice: Choice
    reason: str = Field(min_length=1, max_length=300)


@dataclass(frozen=True)
class JudgedPair:
    """A safe judge outcome for one pair."""

    pair_id: str
    choice: Choice | None
    error_type: str | None = None


@dataclass(frozen=True)
class PairwiseAgreement:
    """Agreement between human and LLM blind decisions."""

    exact_agreement: float
    cohens_kappa: float
    comparable_pair_count: int


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


def summarize_pair_availability(
    legacy_results: Sequence[ContentPipelineCaseResult],
    graph_results: Sequence[ContentPipelineCaseResult],
) -> dict[str, object]:
    """Return safe diagnostics before attempting to create blind pairs."""

    legacy_skus = [result.sku for result in legacy_results]
    graph_skus = [result.sku for result in graph_results]
    if legacy_skus != graph_skus:
        raise ValueError("Legacy and graph results must use the same SKUs.")

    return {
        "completed_pair_count": sum(
            _has_completed_content(legacy) and _has_completed_content(graph)
            for legacy, graph in zip(legacy_results, graph_results, strict=True)
        ),
        "legacy_failures": _failure_type_counts(legacy_results),
        "graph_failures": _failure_type_counts(graph_results),
    }


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

    raw_choices = payload.get("choices", payload.get("pairs"))
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


def judge_blind_pairs(
    pairs: Sequence[BlindContentPair],
    llm_factory: Callable[[], LLMClient],
    progress: Callable[[dict[str, object]], None] | None = None,
) -> list[JudgedPair]:
    """Judge each blind pair independently with a fresh LLM client."""

    prompt_template = (
        Path(__file__).parent / "prompts" / "judge_content_pair.md"
    ).read_text(encoding="utf-8")
    decisions: list[JudgedPair] = []

    for position, pair in enumerate(pairs, start=1):
        event = {"pair_id": pair.pair_id, "position": position, "total": len(pairs)}
        if progress is not None:
            progress({"event_type": "judge_started", **event})
        prompt = prompt_template.format(
            confirmed_attributes=json.dumps(
                dict(pair.confirmed_attributes), ensure_ascii=False
            ),
            card_a=json.dumps(_serialize_visible_card(pair.card_a), ensure_ascii=False),
            card_b=json.dumps(_serialize_visible_card(pair.card_b), ensure_ascii=False),
        )
        try:
            decision = chat_structured(
                client=llm_factory(),
                messages=[Message(role="user", content=prompt)],
                response_schema=PairJudgeDecision,
                max_retries=1,
            )
        except Exception as error:  # noqa: BLE001
            decisions.append(
                JudgedPair(
                    pair_id=pair.pair_id,
                    choice=None,
                    error_type=type(error).__name__,
                )
            )
            if progress is not None:
                progress({"event_type": "judge_error", **event})
            continue

        decisions.append(JudgedPair(pair_id=pair.pair_id, choice=decision.choice))
        if progress is not None:
            progress({"event_type": "judge_completed", **event})

    return decisions


def evaluate_pairwise_agreement(
    human_choices: Sequence[HumanPairChoice],
    judged_pairs: Sequence[JudgedPair],
) -> PairwiseAgreement:
    """Calculate exact agreement and Cohen's kappa for valid judge answers."""

    human_by_id = _choices_by_id(human_choices)
    judged_by_id = _judged_by_id(judged_pairs)
    if set(human_by_id) != set(judged_by_id):
        raise ValueError("Human and judge choices must use the same pair IDs.")

    comparable = [
        (human_by_id[pair_id], judged.choice)
        for pair_id, judged in judged_by_id.items()
        if judged.choice is not None
    ]
    if not comparable:
        return PairwiseAgreement(0.0, 0.0, 0)

    count = len(comparable)
    exact_matches = sum(human == judge for human, judge in comparable)
    labels: tuple[Choice, ...] = ("A", "B", "tie")
    human_counts = {label: sum(human == label for human, _ in comparable) for label in labels}
    judge_counts = {label: sum(judge == label for _, judge in comparable) for label in labels}
    observed = exact_matches / count
    expected = sum(
        human_counts[label] / count * judge_counts[label] / count
        for label in labels
    )
    kappa = 1.0 if observed == expected == 1.0 else (observed - expected) / (1.0 - expected)

    return PairwiseAgreement(observed, kappa, count)


def serialize_pairwise_result(
    pairs: Sequence[BlindContentPair],
    human_choices: Sequence[HumanPairChoice],
    judged_pairs: Sequence[JudgedPair],
) -> dict[str, object]:
    """Serialize only aggregate blind-evaluation results."""

    return {
        "pair_count": len(pairs),
        "human_choices": _choice_counts(human_choices),
        "judge_choices": _choice_counts(
            [
                HumanPairChoice(pair_id=decision.pair_id, choice=decision.choice)
                for decision in judged_pairs
                if decision.choice is not None
            ]
        ),
        "judge_error_types": _error_type_counts(judged_pairs),
    }


def serialize_probe_result(
    result: ContentPipelineCaseResult,
) -> dict[str, object]:
    """Serialize the minimal diagnostic outcome for one pipeline case."""

    return {
        "sku": result.sku,
        "status": result.status,
        "error": result.error,
        "latency_ms": result.latency_ms,
    }


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


def _choices_by_id(
    choices: Sequence[HumanPairChoice],
) -> dict[str, Choice]:
    result = {choice.pair_id: choice.choice for choice in choices}
    if len(result) != len(choices):
        raise ValueError("Human choices must not repeat pair IDs.")
    return result


def _judged_by_id(
    decisions: Sequence[JudgedPair],
) -> dict[str, JudgedPair]:
    result = {decision.pair_id: decision for decision in decisions}
    if len(result) != len(decisions):
        raise ValueError("Judge choices must not repeat pair IDs.")
    return result


def _choice_counts(choices: Sequence[HumanPairChoice]) -> dict[str, int]:
    return {
        label: sum(choice.choice == label for choice in choices)
        for label in ("A", "B", "tie")
    }


def _error_type_counts(
    decisions: Sequence[JudgedPair],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for decision in decisions:
        if decision.error_type is not None:
            counts[decision.error_type] = (
                counts.get(decision.error_type, 0) + 1
            )
    return counts


def _failure_type_counts(
    results: Sequence[ContentPipelineCaseResult],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        if result.status == "manual_review":
            error_type = "manual_review"
        elif result.status == "error":
            error_type = (
                result.error.split(":", maxsplit=1)[0]
                if result.error
                else "unknown_error"
            )
        else:
            continue
        counts[error_type] = counts.get(error_type, 0) + 1
    return counts
