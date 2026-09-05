from typing import Literal, TypedDict

from langgraph.graph import END, START, StateGraph

GraphStatus = Literal["completed", "repair", "manual_review"]


class ContentRoutingState(TypedDict, total=False):
    attempts: int
    violations: list[str]
    status: GraphStatus


def build_content_routing_graph():
    """Build a minimal graph for content-validation routing."""

    graph = StateGraph(ContentRoutingState)

    graph.add_node("validate", _validate)
    graph.add_node("completed", _completed)
    graph.add_node("repair", _repair)
    graph.add_node("manual_review", _manual_review)

    graph.add_edge(START, "validate")
    graph.add_conditional_edges(
        "validate",
        _next_step,
        {
            "completed": "completed",
            "repair": "repair",
            "manual_review": "manual_review",
        },
    )
    graph.add_edge("completed", END)
    graph.add_edge("repair", END)
    graph.add_edge("manual_review", END)

    return graph.compile()


def _validate(state: ContentRoutingState) -> dict[str, object]:
    """Represent validation as one graph node."""

    return {}


def _next_step(state: ContentRoutingState) -> GraphStatus:
    """Choose the next node from validation output."""

    if not state["violations"]:
        return "completed"
    if state["attempts"] < 3:
        return "repair"
    return "manual_review"


def _completed(_: ContentRoutingState) -> ContentRoutingState:
    return {"status": "completed"}


def _repair(_: ContentRoutingState) -> ContentRoutingState:
    return {"status": "repair"}


def _manual_review(_: ContentRoutingState) -> ContentRoutingState:
    return {"status": "manual_review"}