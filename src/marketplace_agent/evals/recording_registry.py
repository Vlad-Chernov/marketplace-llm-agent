from typing import Any

from marketplace_agent.support.agent import ToolRegistry


class RecordingToolRegistry:
    """Обёртка реестра, записывающая вызванные агентом инструменты."""

    def __init__(self, inner_registry: ToolRegistry) -> None:
        self._inner_registry = inner_registry
        self.called_tools: list[str] = []

    def schemas(self) -> list[dict[str, object]]:
        """Вернуть схемы настоящего реестра."""

        return self._inner_registry.schemas()

    def run(
        self,
        name: str,
        arguments: dict[str, object],
    ) -> Any:
        """Записать вызов и передать его настоящему реестру."""

        self.called_tools.append(name)
        return self._inner_registry.run(name, arguments)

    def reset(self) -> None:
        """Очистить список вызовов перед следующим кейсом."""

        self.called_tools.clear()