import json
from pathlib import Path

from marketplace_agent.llm.telemetry import TraceWriter, UsageRecord


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
        },
    )

    event = json.loads(trace_path.read_text(encoding="utf-8"))

    assert event["event_type"] == "llm_call"
    assert event["run_id"] == "run-001"
    assert event["payload"]["model"] == "groq/compound-mini"
    assert "api_key" not in event["payload"]



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