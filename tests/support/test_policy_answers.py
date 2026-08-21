from marketplace_agent.llm.base import FakeLLMClient, LLMResponse
from marketplace_agent.retrieval.documents import PolicyChunk
from marketplace_agent.retrieval.lexical import SearchResult
from marketplace_agent.support.policy_answers import (
    answer_policy_question,
)


class StubRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results

    def search(
        self,
        query: str,
        k: int,
        filters: dict[str, str] | None = None,
    ) -> list[SearchResult]:
        return self._results[:k]


def make_result(chunk_id: str, text: str) -> SearchResult:
    return SearchResult(
        chunk=PolicyChunk(
            chunk_id=chunk_id,
            document_id="returns",
            policy_type="returns",
            heading="Возврат",
            text=text,
        ),
        score=1.0,
        rank=1,
    )


def response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="fake",
        prompt_tokens=0,
        completion_tokens=0,
    )


def test_answers_only_with_returned_policy_citation() -> None:
    answer = answer_policy_question(
        "Сколько дней можно вернуть ноутбук?",
        StubRetriever(
            [
                make_result(
                    "returns-01",
                    "Возврат возможен в течение 14 дней.",
                )
            ]
        ),
        FakeLLMClient(
            [
                response(
                    '{"answer":"Вернуть можно в течение 14 дней.",'
                    '"citations":["returns-01"]}'
                )
            ]
        ),
    )

    assert answer.status == "answered"
    assert answer.answer == "Вернуть можно в течение 14 дней."
    assert answer.citations == ["returns-01"]

def test_returns_no_answer_without_retrieval_or_llm_call() -> None:
    answer = answer_policy_question(
        "Вопрос вне базы",
        StubRetriever([]),
        FakeLLMClient(),
    )

    assert answer.model_dump() == {
        "status": "no_answer",
        "answer": None,
        "citations": [],
    }


def test_rejects_fabricated_citation() -> None:
    answer = answer_policy_question(
        "Сколько дней можно вернуть ноутбук?",
        StubRetriever([make_result("returns-01", "Возврат 14 дней.")]),
        FakeLLMClient(
            [
                response(
                    '{"answer":"Ответ","citations":["warranty-99"]}'
                )
            ]
        ),
    )

    assert answer.model_dump() == {
        "status": "no_answer",
        "answer": None,
        "citations": [],
    }