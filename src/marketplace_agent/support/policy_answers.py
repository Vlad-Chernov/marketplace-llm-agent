import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from marketplace_agent.llm.base import LLMClient, Message
from marketplace_agent.llm.providers import LLMProviderError
from marketplace_agent.llm.structured import (
    StructuredOutputError,
    chat_structured,
)
from marketplace_agent.retrieval.lexical import SearchResult


class Retriever(Protocol):
    """Search policy chunks."""

    def search(
        self,
        query: str,
        k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[SearchResult]: ...


class PolicyAnswer(BaseModel):
    """Store a grounded policy answer or a safe refusal."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["answered", "no_answer"]
    answer: str | None = None
    citations: list[str] = Field(default_factory=list)


class GeneratedPolicyAnswer(BaseModel):
    """Store the structured answer produced by the LLM."""

    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[str]


def answer_policy_question(
    query: str,
    retriever: Retriever,
    llm: LLMClient,
) -> PolicyAnswer:
    """Answer a policy question only from retrieved chunks."""

    if not query.strip():
        return _no_answer()

    results = retriever.search(query, k=3)
    if not results:
        return _no_answer()

    try:
        generated = chat_structured(
            client=llm,
            messages=_build_messages(query, results),
            response_schema=GeneratedPolicyAnswer,
            max_retries=0,
        )
    except (LLMProviderError, StructuredOutputError):
        return _no_answer()

    allowed_ids = {result.chunk.chunk_id for result in results}
    if (
        not generated.answer.strip()
        or not generated.citations
        or not set(generated.citations) <= allowed_ids
    ):
        return _no_answer()

    return PolicyAnswer(
        status="answered",
        answer=generated.answer,
        citations=generated.citations,
    )


def _build_messages(
    query: str,
    results: list[SearchResult],
) -> list[Message]:
    prompt_path = (
        Path(__file__).parent / "prompts" / "answer_policy_question.md"
    )
    prompt = prompt_path.read_text(encoding="utf-8").format(
        query=query,
        chunks=json.dumps(
            [
                {
                    "chunk_id": result.chunk.chunk_id,
                    "heading": result.chunk.heading,
                    "text": result.chunk.text,
                }
                for result in results
            ],
            ensure_ascii=False,
        ),
    )
    return [Message(role="user", content=prompt)]


def _no_answer() -> PolicyAnswer:
    return PolicyAnswer(
        status="no_answer",
        answer=None,
        citations=[],
    )