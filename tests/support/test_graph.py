from marketplace_agent.llm.base import (
    FakeLLMClient,
    LLMResponse,
    Message,
)
from marketplace_agent.support.agent import AgentAnswer
from marketplace_agent.support.graph import build_support_graph
from marketplace_agent.support.tools import ToolResult


class NoCallRegistry:
    def schemas(self) -> list[dict[str, object]]:
        return []

    def run(self, name: str, arguments: dict[str, object]) -> object:
        raise AssertionError(f"Tool must not be called: {name}")


class PolicyRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def schemas(self) -> list[dict[str, object]]:
        return [
            {
                "type": "function",
                "function": {"name": "search_policy"},
            }
        ]

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


def initial_state() -> dict[str, object]:
    return {
        "messages": [Message(role="user", content="Вопрос")],
        "session_id": "session-001",
        "seen_calls": set(),
        "tool_steps": 0,
    }


def test_support_graph_returns_final_answer() -> None:
    graph = build_support_graph(
        registry=NoCallRegistry(),
        llm=FakeLLMClient(
            [
                response(
                    '{"kind":"final","status":"answered",'
                    '"text":"Ответ.","citations":["policy-001"]}'
                )
            ]
        ),
    )

    result = graph.invoke(initial_state())

    assert result["answer"] == AgentAnswer(
        status="answered",
        text="Ответ.",
        citations=["policy-001"],
    )

def test_support_graph_calls_tool_then_returns_answer() -> None:
    registry = PolicyRegistry()
    graph = build_support_graph(
        registry=registry,
        llm=FakeLLMClient(
            [
                response(
                    '{"kind":"tool_call","tool_name":"search_policy",'
                    '"arguments":{"query":"Возврат"}}'
                ),
                response(
                    '{"kind":"final","status":"answered",'
                    '"text":"Возврат возможен 14 дней.",'
                    '"citations":["returns-01"]}'
                ),
            ]
        ),
    )

    result = graph.invoke(initial_state())

    assert registry.calls == [
        ("search_policy", {"query": "Возврат"})
    ]
    assert result["answer"].status == "answered"
    assert result["tool_steps"] == 1

def test_support_graph_escalates_repeated_tool_call() -> None:
    registry = PolicyRegistry()
    graph = build_support_graph(
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

    result = graph.invoke(initial_state())

    assert result["answer"].status == "escalated"
    assert result["answer"].escalation_reason == (
        "Инструмент вызван повторно."
    )
    assert registry.calls == [
        ("search_policy", {"query": "Возврат"})
    ]


def test_support_graph_escalates_after_three_tool_calls() -> None:
    registry = PolicyRegistry()
    graph = build_support_graph(
        registry=registry,
        llm=FakeLLMClient(
            [
                response(
                    '{"kind":"tool_call","tool_name":"search_policy",'
                    '"arguments":{"query":"Возврат"}}'
                ),
                response(
                    '{"kind":"tool_call","tool_name":"search_policy",'
                    '"arguments":{"query":"Гарантия"}}'
                ),
                response(
                    '{"kind":"tool_call","tool_name":"search_policy",'
                    '"arguments":{"query":"Доставка"}}'
                ),
            ]
        ),
    )

    result = graph.invoke(initial_state())

    assert result["answer"].status == "escalated"
    assert result["answer"].escalation_reason == "step_limit"
    assert len(registry.calls) == 3