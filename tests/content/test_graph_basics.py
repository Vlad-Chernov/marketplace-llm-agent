import pytest

from marketplace_agent.content.graph_basics import (
    build_content_routing_graph,
)


@pytest.mark.parametrize(
    ("state", "expected_status"),
    [
        (
            {"attempts": 1, "violations": []},
            "completed",
        ),
        (
            {
                "attempts": 1,
                "violations": ["title-length"],
            },
            "repair",
        ),
        (
            {
                "attempts": 3,
                "violations": ["title-length"],
            },
            "manual_review",
        ),
    ],
)
def test_content_routing_graph_uses_validation_result(
    state: dict[str, object],
    expected_status: str,
) -> None:
    graph = build_content_routing_graph()

    result = graph.invoke(state)

    assert result["status"] == expected_status