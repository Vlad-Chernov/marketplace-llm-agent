from marketplace_agent.llm.base import FakeLLMClient, LLMResponse
from marketplace_agent.support.agent import SupportAgent
from marketplace_agent.support.tools import ToolResult


class NoCallRegistry:
    def schemas(self) -> list[dict[str, object]]:
        return []

    def run(self, name: str, arguments: dict[str, object]) -> object:
        raise AssertionError(f"Инструмент не должен вызываться: {name}")


class PolicyRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def schemas(self) -> list[dict[str, object]]:
        return [{"type": "function", "function": {"name": "search_policy"}}]

    def run(
        self,
        name: str,
        arguments: dict[str, object],
    ) -> ToolResult:
        self.calls.append((name, arguments))
        return ToolResult(
            ok=True,
            data=[
                {
                    "chunk_id": "returns-01",
                    "text": "Возврат возможен в течение 14 дней.",
                }
            ],
            citations=["returns-01"],
        )


def response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="fake",
        prompt_tokens=0,
        completion_tokens=0,
    )


def test_requests_order_number_without_calling_tool() -> None:
    agent = SupportAgent(
        registry=NoCallRegistry(),
        llm=FakeLLMClient(
            [
                response(
                    '{"kind":"final","status":"needs_clarification",'
                    '"text":"Укажите номер заказа.","citations":[]}'
                )
            ]
        ),
    )

    answer = agent.run("Где мой заказ?", "session-001", [])

    assert answer.status == "needs_clarification"
    assert answer.text == "Укажите номер заказа."
    assert answer.citations == []


def test_calls_policy_tool_then_returns_cited_answer() -> None:
    registry = PolicyRegistry()
    agent = SupportAgent(
        registry=registry,
        llm=FakeLLMClient(
            [
                response(
                    '{"kind":"tool_call","tool_name":"search_policy",'
                    '"arguments":{"query":"Срок возврата"}}'
                ),
                response(
                    '{"kind":"final","status":"answered",'
                    '"text":"Вернуть можно в течение 14 дней.",'
                    '"citations":["returns-01"]}'
                ),
            ]
        ),
    )

    answer = agent.run(
        "Сколько дней можно вернуть товар?",
        "session-001",
        [],
    )

    assert registry.calls == [
        ("search_policy", {"query": "Срок возврата"})
    ]
    assert answer.status == "answered"
    assert answer.citations == ["returns-01"]

def test_escalates_before_repeating_identical_tool_call() -> None:
    registry = PolicyRegistry()
    agent = SupportAgent(
        registry=registry,
        llm=FakeLLMClient(
            [
                response(
                    '{"kind":"tool_call","tool_name":"search_policy",'
                    '"arguments":{"query":"Возврат"}}'
                ),
                response(
                    '{"kind":"tool_call","tool_name":"search_policy",'
                    '"arguments":{"query":"Возврат"}}'
                ),
            ]
        ),
    )

    answer = agent.run("Вопрос", "session-001", [])

    assert answer.status == "escalated"
    assert answer.escalation_reason == "Инструмент вызван повторно."
    assert registry.calls == [
        ("search_policy", {"query": "Возврат"})
    ]