import json
from pathlib import Path

from marketplace_agent.llm.base import FakeLLMClient, LLMResponse, Message
from marketplace_agent.llm.telemetry import (
    TraceWriter,
    TracingLLMClient,
    UsageRecord,
    trace_event,
    trace_run,
)


def test_trace_writer_records_event_without_secrets(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    writer = TraceWriter(trace_path)

    writer.write(
        event_type="llm_call",
        run_id="run-001",
        payload={
            "model": "groq/compound-mini",
            "prompt_tokens": 10,
            "api_key": "secret-value",
            "gigachat_authorization_key": "gigachat-secret",
            "access_token": "access-secret",
        },
    )

    event = json.loads(trace_path.read_text(encoding="utf-8"))

    assert event["event_type"] == "llm_call"
    assert event["run_id"] == "run-001"
    assert event["payload"]["model"] == "groq/compound-mini"
    assert "api_key" not in event["payload"]
    assert "gigachat_authorization_key" not in event["payload"]
    assert "access_token" not in event["payload"]



def test_usage_record_calculates_estimated_cost() -> None:
    usage = UsageRecord(
        model="test-model",
        prompt_tokens=1_000,
        completion_tokens=500,
        input_price_per_million=1.0,
        output_price_per_million=2.0,
        latency_ms=250,
    )

    assert usage.estimated_cost_usd == 0.002
    assert usage.latency_ms == 250

def test_trace_writer_redacts_nested_pii(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    writer = TraceWriter(trace_path)

    writer.write(
        event_type="llm_call",
        run_id="run-002",
        payload={
            "message": "Анна Петрова, позвоните +7 900 111-22-33.",
            "context": {
                "address": "улица Ленина, дом 10",
                "email": "anna@example.com",
            },
        },
    )

    saved_text = trace_path.read_text(encoding="utf-8")

    assert "Анна Петрова" not in saved_text
    assert "+7 900 111-22-33" not in saved_text
    assert "улица Ленина, дом 10" not in saved_text
    assert "anna@example.com" not in saved_text
    assert "[PERSON]" in saved_text
    assert "[PHONE]" in saved_text
    assert "[ADDRESS]" in saved_text
    assert "[EMAIL]" in saved_text


def test_trace_run_saves_events_and_optionally_prints_them(
    tmp_path: Path,
) -> None:
    terminal_lines: list[str] = []
    trace_path = tmp_path / "trace.jsonl"
    writer = TraceWriter(
        trace_path,
        terminal_output=terminal_lines.append,
    )

    with trace_run(writer, "run-verbose"):
        trace_event(
            "custom_step",
            {
                "message": "Позвоните +7 900 111-22-33.",
                "authorization": "secret-value",
            },
        )

    saved_text = trace_path.read_text(encoding="utf-8")
    events = [json.loads(line) for line in saved_text.splitlines()]

    assert [event["event_type"] for event in events] == [
        "run_started",
        "custom_step",
        "run_completed",
    ]
    assert terminal_lines == [
        "[trace] run_started",
        "[trace] custom_step",
        "[trace] run_completed",
    ]
    assert "secret-value" not in saved_text
    assert "+7 900 111-22-33" not in saved_text


def test_tracing_llm_client_records_request_and_result(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    writer = TraceWriter(trace_path)
    client = TracingLLMClient(
        FakeLLMClient(
            chat_responses=[
                LLMResponse(
                    content='{"answer":"ok"}',
                    model="fake-model",
                    prompt_tokens=12,
                    completion_tokens=4,
                )
            ]
        )
    )

    with trace_run(writer, "run-llm"):
        client.chat(
            messages=[Message(role="user", content="Проверь карточку")],
            tools=None,
            response_schema=None,
            temperature=0.0,
            max_tokens=100,
        )

    events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]

    assert [event["event_type"] for event in events] == [
        "run_started",
        "llm_call_started",
        "llm_call_completed",
        "run_completed",
    ]
    assert events[1]["payload"]["messages"] == [
        {"role": "user", "content": "Проверь карточку"}
    ]
    assert events[2]["payload"]["model"] == "fake-model"
    assert events[2]["payload"]["prompt_tokens"] == 12
    assert events[2]["payload"]["completion_tokens"] == 4
    assert events[2]["payload"]["cache_hit"] is False
    assert events[2]["payload"]["latency_ms"] >= 0
