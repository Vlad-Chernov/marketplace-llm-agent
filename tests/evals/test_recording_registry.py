from marketplace_agent.evals.recording_registry import (
    RecordingToolRegistry,
)
from marketplace_agent.support.tools import ToolResult


class FakeRegistry:
    def schemas(self) -> list[dict[str, object]]:
        return [{"type": "function", "function": {"name": "search_policy"}}]

    def run(
        self,
        name: str,
        arguments: dict[str, object],
    ) -> ToolResult:
        assert name == "search_policy"
        assert arguments == {"query": "Возврат"}

        return ToolResult(
            ok=True,
            data=[],
            citations=["returns-01"],
        )


def test_records_called_tool_names() -> None:
    registry = RecordingToolRegistry(FakeRegistry())

    result = registry.run("search_policy", {"query": "Возврат"})

    assert result.ok is True
    assert registry.called_tools == ["search_policy"]

    registry.reset()

    assert registry.called_tools == []