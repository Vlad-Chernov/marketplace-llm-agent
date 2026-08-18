from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

REQUIRED_ATTRIBUTES = {
    "screen_diagonal_in",
    "processor",
    "ram_gb",
    "storage_gb",
}

FILTERABLE_ATTRIBUTES = {
    "screen_diagonal_in",
    "screen_resolution",
    "processor",
    "ram_gb",
    "storage_gb",
    "graphics_card",
    "operating_system",
    "weight_kg",
    "color",
    "keyboard_layout",
    "refresh_rate_hz",
}


@dataclass(frozen=True)
class AttributeScore:
    true_positive: int
    false_positive: int
    false_negative: int

    @property
    def precision(self) -> float:
        denominator = self.true_positive + self.false_positive
        return self.true_positive / denominator if denominator else 0.0

    @property
    def recall(self) -> float:
        denominator = self.true_positive + self.false_negative
        return self.true_positive / denominator if denominator else 0.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0


@dataclass(frozen=True)
class AttributeMetrics:
    overall: AttributeScore
    required: AttributeScore
    filterable: AttributeScore
    hallucination_count: int
    prediction_count: int

    @property
    def hallucination_rate(self) -> float:
        return (
            self.hallucination_count / self.prediction_count
            if self.prediction_count
            else 0.0
        )


def evaluate_attribute_extraction(
    predictions: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
) -> AttributeMetrics:
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length.")

    overall, hallucinations, prediction_count = _score(
        predictions,
        references,
        attribute_keys=None,
    )
    required, _, _ = _score(
        predictions,
        references,
        attribute_keys=REQUIRED_ATTRIBUTES,
    )
    filterable, _, _ = _score(
        predictions,
        references,
        attribute_keys=FILTERABLE_ATTRIBUTES,
    )

    return AttributeMetrics(
        overall=overall,
        required=required,
        filterable=filterable,
        hallucination_count=hallucinations,
        prediction_count=prediction_count,
    )


def _score(
    predictions: Sequence[Mapping[str, Any]],
    references: Sequence[Mapping[str, Any]],
    attribute_keys: set[str] | None,
) -> tuple[AttributeScore, int, int]:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    hallucination_count = 0
    prediction_count = 0

    for prediction, reference in zip(predictions, references, strict=True):
        keys = set(prediction) | set(reference)
        if attribute_keys is not None:
            keys &= attribute_keys

        for key in keys:
            predicted_value = prediction.get(key)
            reference_value = reference.get(key)
            has_prediction = predicted_value is not None
            has_reference = reference_value is not None

            if has_prediction:
                prediction_count += 1

            if has_prediction and has_reference:
                if predicted_value == reference_value:
                    true_positive += 1
                else:
                    false_positive += 1
                    false_negative += 1
            elif has_prediction:
                false_positive += 1
                hallucination_count += 1
            elif has_reference:
                false_negative += 1

    return (
        AttributeScore(
            true_positive=true_positive,
            false_positive=false_positive,
            false_negative=false_negative,
        ),
        hallucination_count,
        prediction_count,
    )