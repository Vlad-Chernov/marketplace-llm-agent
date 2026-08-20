import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from marketplace_agent.retrieval.documents import PolicyChunk

K1 = 1.5
B = 0.75
TOKEN_PATTERN = re.compile(r"[a-zа-я0-9]+")


@dataclass(frozen=True)
class SearchResult:
    """Store one ranked policy chunk."""

    chunk: PolicyChunk
    score: float
    rank: int


class BM25Retriever:
    """Search policy chunks using an in-memory BM25 index."""

    def __init__(self, chunks: Sequence[PolicyChunk]) -> None:
        self._chunks = list(chunks)
        self._term_frequencies = [
            Counter(_normalize(chunk.text)) for chunk in self._chunks
        ]
        self._document_lengths = [
            sum(frequencies.values())
            for frequencies in self._term_frequencies
        ]
        self._average_document_length = (
            sum(self._document_lengths) / len(self._document_lengths)
            if self._document_lengths
            else 0.0
        )
        self._document_frequencies = Counter(
            term
            for frequencies in self._term_frequencies
            for term in frequencies
        )

    def search(
        self,
        query: str,
        k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[SearchResult]:
        """Return the highest-scoring chunks for a query."""

        query_terms = _normalize(query)
        if not query_terms or k <= 0:
            return []

        scored_chunks = [
            (chunk, self._score(index, query_terms))
            for index, chunk in enumerate(self._chunks)
            if _matches_filters(chunk, filters)
        ]
        ranked_chunks = sorted(
            (
                (chunk, score)
                for chunk, score in scored_chunks
                if score > 0
            ),
            key=lambda item: (-item[1], item[0].chunk_id),
        )
        return [
            SearchResult(chunk=chunk, score=score, rank=rank)
            for rank, (chunk, score) in enumerate(ranked_chunks[:k], start=1)
        ]

    def _score(self, index: int, query_terms: list[str]) -> float:
        if self._average_document_length == 0:
            return 0.0

        frequencies = self._term_frequencies[index]
        document_length = self._document_lengths[index]
        score = 0.0

        for term in set(query_terms):
            term_frequency = frequencies[term]
            if term_frequency == 0:
                continue

            document_frequency = self._document_frequencies[term]
            inverse_document_frequency = math.log(
                1
                + (len(self._chunks) - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            denominator = term_frequency + K1 * (
                1 - B + B * document_length / self._average_document_length
            )
            score += inverse_document_frequency * (
                term_frequency * (K1 + 1) / denominator
            )

        return score


def _normalize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold().replace("ё", "е"))


def _matches_filters(
    chunk: PolicyChunk,
    filters: Mapping[str, str] | None,
) -> bool:
    if filters is None:
        return True

    return all(
        key in {"document_id", "policy_type"}
        and getattr(chunk, key) == value
        for key, value in filters.items()
    )