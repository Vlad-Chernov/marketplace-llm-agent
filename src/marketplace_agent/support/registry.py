from collections.abc import Sequence
from typing import Any

from marketplace_agent.llm.telemetry import trace_event
from marketplace_agent.support.tools import Tool, ToolResult


class ToolRegistry:
    """Store support tools and expose their schemas to an LLM."""

    def __init__(self, tools: Sequence[Tool]) -> None:
        self._tools: dict[str, Tool] = {}

        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"Duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool

    def run(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        trace_event(
            "tool_call_started",
            {"name": name, "arguments": arguments},
        )
        tool = self._tools.get(name)
        if tool is None:
            return _trace_result(
                name,
                ToolResult(
                    ok=False,
                    error="Инструмент не найден.",
                    error_code="unknown_tool",
                ),
            )

        try:
            tool.input_model.model_validate(arguments)
        except (TypeError, ValueError):
            return _trace_result(
                name,
                ToolResult(
                    ok=False,
                    error="Некорректные аргументы инструмента.",
                    error_code="invalid_arguments",
                ),
            )

        return _trace_result(name, tool.run(**arguments))

    def schemas(self) -> list[dict[str, object]]:
        """Return OpenAI-compatible function schemas."""

        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_model.model_json_schema(),
                },
            }
            for tool in self._tools.values()
        ]


def _trace_result(name: str, result: ToolResult) -> ToolResult:
    trace_event(
        "tool_call_completed",
        {"name": name, "result": result.model_dump(mode="json")},
    )
    return result
