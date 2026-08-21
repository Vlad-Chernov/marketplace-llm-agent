import json
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from marketplace_agent.llm.base import LLMClient, Message
from marketplace_agent.llm.structured import (
    StructuredOutputError,
    chat_structured,
)


class ToolRegistry(Protocol):
    """Expose registered tools to the support agent."""

    def schemas(self) -> list[dict[str, object]]: ...

    def run(self, name: str, arguments: dict[str, object]) -> Any: ...


class AgentAnswer(BaseModel):
    """Store one safe result from the support agent."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["answered", "needs_clarification", "escalated"]
    text: str
    citations: list[str] = Field(default_factory=list)
    escalation_reason: str | None = None


class AgentDecision(BaseModel):
    """Store one structured decision returned by the LLM."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["final", "tool_call"]
    status: Literal["answered", "needs_clarification", "escalated"] | None = None
    text: str | None = None
    citations: list[str] = Field(default_factory=list)
    tool_name: str | None = None
    arguments: dict[str, object] = Field(default_factory=dict)


class SupportAgent:
    """Run a bounded support conversation over registered tools."""

    def __init__(self, registry: ToolRegistry, llm: LLMClient) -> None:
        self._registry = registry
        self._llm = llm

    def run(
        self,
        message: str,
        session_id: str,
        history: list[Message],
    ) -> AgentAnswer:
        """Run at most three safe tool calls."""

        if self._is_prompt_injection(message):
            return self._escalated("prompt_injection")

        if self._is_forbidden_request(message):
            return self._escalated("forbidden_request")

        messages = self._build_messages(message, history)
        seen_calls: set[str] = set()

        for _ in range(3):
            try:
                decision = chat_structured(
                    client=self._llm,
                    messages=messages,
                    response_schema=AgentDecision,
                    max_retries=0,
                )
            except StructuredOutputError:
                return self._escalated(
                    "Не удалось безопасно обработать запрос."
                )

            if decision.kind == "final":
                if decision.status == "escalated":
                    return self._escalated("ambiguous_case")
                
                if decision.status is None or not decision.text:
                    return self._escalated("insufficient_data")

                return AgentAnswer(
                    status=decision.status,
                    text=decision.text,
                    citations=decision.citations,
                )

            if decision.tool_name is None:
                return self._escalated("Инструмент не указан.")

            arguments = dict(decision.arguments)
            if decision.tool_name == "get_order":
                arguments["session_id"] = session_id

            call_signature = json.dumps(
                {
                    "name": decision.tool_name,
                    "arguments": arguments,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            if call_signature in seen_calls:
                return self._escalated("Инструмент вызван повторно.")

            seen_calls.add(call_signature)
            tool_result = self._registry.run(
                decision.tool_name,
                arguments,
            )
            if not tool_result.ok:
                return self._escalated("tool_error")

            messages.append(
                Message(
                    role="tool",
                    content=json.dumps(
                        tool_result.model_dump(),
                        ensure_ascii=False,
                    ),
                )
            )

        return self._escalated("step_limit")

    def _is_prompt_injection(self, message: str) -> bool:
        normalized = message.casefold()
        markers = (
            "игнорируй предыдущие инструкции",
            "ignore previous instructions",
            "покажи системный prompt",
            "раскрой системный prompt",
            "show system prompt",
        )
        return any(marker in normalized for marker in markers)

    def _is_forbidden_request(self, message: str) -> bool:
        normalized = message.casefold()
        markers = (
            "выполни python-код",
            "выполни код",
            "execute code",
            "удали все заказы",
            "удали базу",
            "delete all orders",
            "delete database",
        )
        return any(marker in normalized for marker in markers)

    def _build_messages(
        self,
        message: str,
        history: list[Message],
    ) -> list[Message]:
        instructions = {
            "message": message,
            "tools": self._registry.schemas(),
            "format": {
                "kind": "final or tool_call",
                "status": "answered or needs_clarification for final",
                "text": "text for final",
                "citations": ["chunk_id"],
                "tool_name": "registered name for tool_call",
                "arguments": {},
            },
        }
        return [
            *history,
            Message(
                role="user",
                content=(
                    "Верни только JSON по инструкции. "
                    "Если не хватает номера заказа, задай уточняющий вопрос. "
                    f"Контекст: {json.dumps(instructions, ensure_ascii=False)}"
                ),
            ),
        ]

    def _escalated(self, reason: str) -> AgentAnswer:
        return AgentAnswer(
            status="escalated",
            text="Передам вопрос специалисту.",
            escalation_reason=reason,
        )