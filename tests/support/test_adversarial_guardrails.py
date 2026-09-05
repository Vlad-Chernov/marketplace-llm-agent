import json
from pathlib import Path

from marketplace_agent.llm.base import FakeLLMClient, LLMResponse
from marketplace_agent.support.agent import SupportAgent

ADVERSARIAL_CASES_PATH = Path(
    "data/gold/adversarial_cases.json"
)


class NoCallRegistry:
    def schemas(self) -> list[dict[str, object]]:
        return []

    def run(self, name: str, arguments: dict[str, object]) -> object:
        raise AssertionError(f"Инструмент не должен вызываться: {name}")


class RecordingLLMClient(FakeLLMClient):
    def __init__(self) -> None:
        super().__init__(
            [
                LLMResponse(
                    content=(
                        '{"kind":"final",'
                        '"status":"needs_clarification",'
                        '"text":"Уточните вопрос.",'
                        '"citations":[]}'
                    ),
                    model="fake",
                    prompt_tokens=0,
                    completion_tokens=0,
                )
            ]
        )
        self.calls = 0

    def chat(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls += 1
        return super().chat(**kwargs)


def test_all_adversarial_cases_escalate_before_llm_or_tool() -> None:
    raw_cases = json.loads(
        ADVERSARIAL_CASES_PATH.read_text(encoding="utf-8")
    )

    for case in raw_cases:
        client = RecordingLLMClient()
        answer = SupportAgent(
            registry=NoCallRegistry(),
            llm=client,
        ).run(
            message=case["input"]["message"],
            session_id=case["input"]["session_id"],
            history=[],
        )

        assert answer.status == "escalated", case["id"]
        assert client.calls == 0, case["id"]