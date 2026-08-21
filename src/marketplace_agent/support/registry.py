from collections.abc import Sequence
from typing import Any

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
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                ok=False,
                error="Инструмент не найден.",
                error_code="unknown_tool",
            )

        try:
            tool.input_model.model_validate(arguments)
        except (TypeError, ValueError):
            return ToolResult(
                ok=False,
                error="Некорректные аргументы инструмента.",
                error_code="invalid_arguments",
            )

        return tool.run(**arguments)

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