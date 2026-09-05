import json
from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from marketplace_agent.llm.base import LLMClient, Message
from marketplace_agent.llm.structured import (
    StructuredOutputError,
    chat_structured,
)
from marketplace_agent.support.agent import (
    AgentAnswer,
    AgentDecision,
    ToolRegistry,
)


class SupportGraphState(TypedDict, total=False):
    messages: list[Message]
    session_id: str
    seen_calls: set[str]
    tool_steps: int
    decision: AgentDecision
    answer: AgentAnswer
    escalation_reason: str


def build_support_graph(
    registry: ToolRegistry,
    llm: LLMClient,
):
    """Build the safe support-agent graph."""

    graph = StateGraph(SupportGraphState)
    graph.add_node("decide", _decide(llm))
    graph.add_node("execute_tool", _execute_tool(registry))
    graph.add_node("finalize", _finalize)
    graph.add_node("step_limit", _step_limit)
    graph.add_node("escalate", _escalate)

    graph.add_edge(START, "decide")
    graph.add_conditional_edges(
        "decide",
        _route_after_decision,
        {
            "finalize": "finalize",
            "execute_tool": "execute_tool",
            "escalate": "escalate",
        },
    )
    graph.add_conditional_edges(
        "execute_tool",
        _route_after_tool,
        {
            "decide": "decide",
            "step_limit": "step_limit",
            "escalate": "escalate",
        },
    )
    graph.add_edge("finalize", END)
    graph.add_edge("step_limit", "escalate")
    graph.add_edge("escalate", END)

    return graph.compile()


def _decide(llm: LLMClient):
    def decide(
        state: SupportGraphState,
    ) -> dict[str, object]:
        try:
            decision = chat_structured(
                client=llm,
                messages=state["messages"],
                response_schema=AgentDecision,
                max_retries=0,
            )
        except StructuredOutputError:
            return {
                "escalation_reason": (
                    "Не удалось безопасно обработать запрос."
                )
            }

        return {"decision": decision}

    return decide


def _route_after_decision(
    state: SupportGraphState,
) -> Literal["finalize", "execute_tool", "escalate"]:
    if "escalation_reason" in state:
        return "escalate"
    if state["decision"].kind == "final":
        return "finalize"
    return "execute_tool"


def _execute_tool(registry: ToolRegistry):
    def execute_tool(
        state: SupportGraphState,
    ) -> dict[str, object]:
        decision = state["decision"]

        if decision.tool_name is None:
            return {"escalation_reason": "Инструмент не указан."}

        arguments = dict(decision.arguments)
        if decision.tool_name == "get_order":
            arguments["session_id"] = state["session_id"]

        call_signature = json.dumps(
            {
                "name": decision.tool_name,
                "arguments": arguments,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        if call_signature in state["seen_calls"]:
            return {
                "escalation_reason": "Инструмент вызван повторно."
            }

        tool_result = registry.run(decision.tool_name, arguments)
        if not tool_result.ok:
            return {"escalation_reason": "tool_error"}

        return {
            "messages": [
                *state["messages"],
                Message(
                    role="user",
                    content=(
                        "Результат инструмента. Используй только эти "
                        "данные для следующего JSON-ответа:\n"
                        f"{json.dumps(tool_result.model_dump(), ensure_ascii=False)}"
                    ),
                ),
            ],
            "seen_calls": {
                *state["seen_calls"],
                call_signature,
            },
            "tool_steps": state["tool_steps"] + 1,
        }

    return execute_tool


def _route_after_tool(
    state: SupportGraphState,
) -> Literal["decide", "step_limit", "escalate"]:
    if "escalation_reason" in state:
        return "escalate"
    if state["tool_steps"] == 3:
        return "step_limit"
    return "decide"


def _finalize(
    state: SupportGraphState,
) -> dict[str, AgentAnswer]:
    decision = state["decision"]

    if decision.status == "escalated":
        return {
            "answer": _escalated_answer("ambiguous_case"),
        }

    if decision.status is None or not decision.text:
        return {
            "answer": _escalated_answer("insufficient_data"),
        }

    return {
        "answer": AgentAnswer(
            status=decision.status,
            text=decision.text,
            citations=decision.citations,
        )
    }


def _step_limit(
    _: SupportGraphState,
) -> dict[str, str]:
    return {"escalation_reason": "step_limit"}


def _escalate(
    state: SupportGraphState,
) -> dict[str, AgentAnswer]:
    return {
        "answer": _escalated_answer(
            state["escalation_reason"],
        )
    }


def _escalated_answer(reason: str) -> AgentAnswer:
    return AgentAnswer(
        status="escalated",
        text="Передам вопрос специалисту.",
        escalation_reason=reason,
    )