import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from marketplace_agent.privacy.pii import PiiRedactor

SENSITIVE_KEYS = {"api_key", "authorization", "token", "password"}


@dataclass(frozen=True)
class UsageRecord:
    """Store token usage, latency, and estimated call cost."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    input_price_per_million: float
    output_price_per_million: float
    latency_ms: int

    @property
    def estimated_cost_usd(self) -> float:
        """Return the estimated cost in US dollars."""

        input_cost = (
            self.prompt_tokens / 1_000_000 * self.input_price_per_million
        )
        output_cost = (
            self.completion_tokens
            / 1_000_000
            * self.output_price_per_million
        )
        return input_cost + output_cost


class TraceWriter:
    """Write sanitized trace events to a JSONL file."""

    def __init__(self, trace_path: Path) -> None:
        self.trace_path = trace_path

    def write(
        self,
        event_type: str,
        run_id: str,
        payload: dict[str, Any],
    ) -> None:
        """Append one sanitized event to the trace file."""

        self.trace_path.parent.mkdir(parents=True, exist_ok=True)

        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": event_type,
            "run_id": run_id,
            "payload": self._sanitize_payload(payload),
        }

        with self.trace_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _sanitize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        with PiiRedactor() as redactor:
            sanitized = self._sanitize_value(payload, redactor)

        assert isinstance(sanitized, dict)
        return sanitized

    def _sanitize_value(
        self,
        value: Any,
        redactor: PiiRedactor,
    ) -> Any:
        if isinstance(value, str):
            return redactor.redact(value)
        if isinstance(value, dict):
            return {
                key: self._sanitize_value(item, redactor)
                for key, item in value.items()
                if key.lower() not in SENSITIVE_KEYS
            }
        if isinstance(value, list | tuple):
            return [
                self._sanitize_value(item, redactor)
                for item in value
            ]
        return value