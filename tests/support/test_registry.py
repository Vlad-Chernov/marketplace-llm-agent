import json
from pathlib import Path

from marketplace_agent.llm.telemetry import TraceWriter, trace_run
from marketplace_agent.support.registry import ToolRegistry
from marketplace_agent.support.tools import ToolResult


class EchoInput:
    @classmethod
    def model_json_schema(cls) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
            },
            "required": ["text"],
        }

    @classmethod
    def model_validate(cls, arguments: dict[str, object]) -> object:
        if not isinstance(arguments.get("text"), str):
            raise TypeError("text is required")
        return object()


class EchoTool:
    name = "echo"
    description = "Вернуть переданный текст."
    input_model = EchoInput

    def run(self, **arguments: object) -> ToolResult:
        return ToolResult(ok=True, data={"text": arguments["text"]})


def test_runs_registered_tool_and_generates_schema() -> None:
    registry = ToolRegistry([EchoTool()])

    result = registry.run("echo", {"text": "Привет"})

    assert result.ok is True
    assert registry.schemas() == [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "Вернуть переданный текст.",
                "parameters": EchoInput.model_json_schema(),
            },
        }
    ]


def test_returns_safe_errors_for_unknown_or_invalid_tool_call() -> None:
    registry = ToolRegistry([EchoTool()])

    assert registry.run("missing", {}).error_code == "unknown_tool"
    assert registry.run("echo", {}).error_code == "invalid_arguments"


def test_records_tool_call_and_result_in_active_trace(
    tmp_path: Path,
) -> None:
    trace_path = tmp_path / "trace.jsonl"
    registry = ToolRegistry([EchoTool()])

    with trace_run(TraceWriter(trace_path), "run-tool"):
        result = registry.run("echo", {"text": "Привет"})

    events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]

    assert result.ok is True
    assert [event["event_type"] for event in events] == [
        "run_started",
        "tool_call_started",
        "tool_call_completed",
        "run_completed",
    ]
    assert events[1]["payload"] == {
        "name": "echo",
        "arguments": {"text": "Привет"},
    }
    assert events[2]["payload"]["name"] == "echo"
    assert events[2]["payload"]["result"]["ok"] is True
