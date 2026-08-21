from collections.abc import Mapping
from typing import Protocol

from marketplace_agent.retrieval.lexical import SearchResult


class Retriever(Protocol):
    """Search policy chunks."""

    def search(
        self,
        query: str,
        k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[SearchResult]: ...


class HybridRetriever:
    """Combine lexical and vector results with reciprocal rank fusion."""

    def __init__(
        self,
        lexical: Retriever,
        vector: Retriever,
        rrf_k: int = 60,
        candidate_k: int = 10,
    ) -> None:
        self._lexical = lexical
        self._vector = vector
        self._rrf_k = rrf_k
        self._candidate_k = candidate_k

    def search(
        self,
        query: str,
        k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[SearchResult]:
        """Return hybrid results ordered by reciprocal rank fusion."""

        if not query.strip() or k <= 0:
            return []

        scores: dict[str, float] = {}
        chunks = {}

        for retriever in (self._lexical, self._vector):
            for result in retriever.search(
                query,
                self._candidate_k,
                filters,
            ):
                chunk_id = result.chunk.chunk_id
                chunks[chunk_id] = result.chunk
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (
                    self._rrf_k + result.rank
                )

        ranked_chunk_ids = sorted(
            scores,
            key=lambda chunk_id: (-scores[chunk_id], chunk_id),
        )

        return [
            SearchResult(
                chunk=chunks[chunk_id],
                score=scores[chunk_id],
                rank=rank,
            )
            for rank, chunk_id in enumerate(ranked_chunk_ids[:k], start=1)
        ]