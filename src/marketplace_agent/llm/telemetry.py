import json
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

from pydantic import BaseModel

from marketplace_agent.llm.base import LLMClient, LLMResponse, Message
from marketplace_agent.privacy.pii import PiiRedactor

SENSITIVE_KEY_PARTS = {
    "api_key",
    "apikey",
    "authorization",
    "password",
    "secret",
    "token",
}


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
    """Write sanitized trace events to JSONL and optionally the terminal."""

    def __init__(
        self,
        trace_path: Path,
        terminal_output: Callable[[str], None] | None = None,
    ) -> None:
        self.trace_path = trace_path
        self._terminal_output = terminal_output
        self._redactor = PiiRedactor()

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

        if self._terminal_output is not None:
            self._terminal_output(f"[trace] {event_type}")

    def close(self) -> None:
        """Release resources used by PII redaction."""

        self._redactor.close()

    def _sanitize_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        sanitized = self._sanitize_value(payload, self._redactor)

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
                if not _is_sensitive_key(key)
            }
        if isinstance(value, list | tuple):
            return [
                self._sanitize_value(item, redactor)
                for item in value
            ]
        return value


_active_trace: ContextVar[tuple[TraceWriter, str] | None] = ContextVar(
    "active_trace",
    default=None,
)


@contextmanager
def trace_run(writer: TraceWriter, run_id: str) -> Iterator[None]:
    """Activate one trace for all nested application calls."""

    token = _active_trace.set((writer, run_id))
    writer.write("run_started", run_id, {})
    try:
        yield
    except Exception as error:
        writer.write(
            "run_failed",
            run_id,
            {
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    else:
        writer.write("run_completed", run_id, {})
    finally:
        _active_trace.reset(token)
        writer.close()


def trace_event(event_type: str, payload: dict[str, Any]) -> None:
    """Write an event when execution is inside an active trace."""

    active_trace = _active_trace.get()
    if active_trace is None:
        return

    writer, run_id = active_trace
    writer.write(event_type, run_id, payload)


def create_trace_writer(
    trace_directory: Path,
    run_id: str,
) -> TraceWriter:
    """Create the standard file writer for one application run."""

    terminal_output = print if os.getenv("TRACE_VERBOSE") == "1" else None
    return TraceWriter(
        trace_directory / f"{run_id}.jsonl",
        terminal_output=terminal_output,
    )


class TracingLLMClient:
    """Record LLM requests, responses and failures in the active trace."""

    def __init__(
        self,
        inner_client: LLMClient,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        self._inner_client = inner_client
        self._clock = clock

    @property
    def last_call_was_cache_hit(self) -> bool:
        """Expose cache state to metering wrappers."""

        return bool(
            getattr(self._inner_client, "last_call_was_cache_hit", False)
        )

    def chat(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None,
        response_schema: type[BaseModel] | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        """Call the wrapped client and trace the observable request."""

        trace_event(
            "llm_call_started",
            {
                "messages": [message.model_dump() for message in messages],
                "tool_names": _tool_names(tools),
                "response_schema": (
                    response_schema.__name__
                    if response_schema is not None
                    else None
                ),
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        started_at = self._clock()
        try:
            response = self._inner_client.chat(
                messages=messages,
                tools=tools,
                response_schema=response_schema,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as error:
            trace_event(
                "llm_call_failed",
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "latency_ms": round(
                        (self._clock() - started_at) * 1000
                    ),
                },
            )
            raise

        trace_event(
            "llm_call_completed",
            {
                "model": response.model,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "cache_hit": self.last_call_was_cache_hit,
                "latency_ms": round(
                    (self._clock() - started_at) * 1000
                ),
                "response": response.content,
            },
        )
        return response

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Delegate embeddings and trace their count without storing texts."""

        trace_event("embedding_started", {"text_count": len(texts)})
        started_at = self._clock()
        try:
            vectors = self._inner_client.embed(texts)
        except Exception as error:
            trace_event(
                "embedding_failed",
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            raise
        trace_event(
            "embedding_completed",
            {
                "text_count": len(texts),
                "vector_count": len(vectors),
                "latency_ms": round(
                    (self._clock() - started_at) * 1000
                ),
            },
        )
        return vectors


def _tool_names(
    tools: list[dict[str, Any]] | None,
) -> list[str]:
    if tools is None:
        return []

    return [
        str(tool.get("function", {}).get("name", "unknown"))
        for tool in tools
    ]


def _is_sensitive_key(key: str) -> bool:
    normalized = key.casefold().replace("-", "_")
    parts = set(normalized.split("_"))
    return (
        normalized in SENSITIVE_KEY_PARTS
        or normalized.endswith("_api_key")
        or bool(
            parts
            & {"authorization", "password", "secret", "token"}
        )
    )
