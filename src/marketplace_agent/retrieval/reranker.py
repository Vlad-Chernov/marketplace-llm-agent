from collections.abc import Mapping, Sequence
from typing import Protocol

from sentence_transformers import CrossEncoder

from marketplace_agent.retrieval.lexical import SearchResult

MODEL_NAME = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class PairScorer(Protocol):
    """Assign a relevance score to every query and text pair."""

    def score(self, pairs: list[tuple[str, str]]) -> list[float]: ...


class Reranker(Protocol):
    """Rerank existing retrieval candidates."""

    def rerank(
        self,
        query: str,
        results: Sequence[SearchResult],
        k: int,
    ) -> list[SearchResult]: ...

class Retriever(Protocol):
    """Search policy chunks."""

    def search(
        self,
        query: str,
        k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[SearchResult]: ...


class SentenceTransformerCrossEncoder:
    """Score query and policy-chunk pairs with a cross-encoder."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model_name = model_name
        self._model: CrossEncoder | None = None

    def score(self, pairs: list[tuple[str, str]]) -> list[float]:
        if self._model is None:
            self._model = CrossEncoder(self._model_name)

        return [float(score) for score in self._model.predict(pairs)]


class CrossEncoderReranker:
    """Rerank retriever candidates by cross-encoder relevance."""

    def __init__(self, scorer: PairScorer | None = None) -> None:
        self._scorer = scorer or SentenceTransformerCrossEncoder()

    def rerank(
        self,
        query: str,
        results: Sequence[SearchResult],
        k: int,
    ) -> list[SearchResult]:
        if not query.strip() or not results or k <= 0:
            return []

        scores = self._scorer.score(
            [(query, result.chunk.text) for result in results]
        )
        ranked = sorted(
            zip(results, scores, strict=True),
            key=lambda item: (-float(item[1]), item[0].chunk.chunk_id),
        )

        return [
            SearchResult(
                chunk=result.chunk,
                score=float(score),
                rank=rank,
            )
            for rank, (result, score) in enumerate(ranked[:k], start=1)
        ]

class RerankedRetriever:
    """Rerank a fixed number of candidates from another retriever."""

    def __init__(
        self,
        base_retriever: Retriever,
        reranker: Reranker,
        candidate_k: int = 10,
    ) -> None:
        self._base_retriever = base_retriever
        self._reranker = reranker
        self._candidate_k = candidate_k

    def search(
        self,
        query: str,
        k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[SearchResult]:
        if not query.strip() or k <= 0:
            return []

        candidates = self._base_retriever.search(
            query,
            self._candidate_k,
            filters,
        )
        return self._reranker.rerank(query, candidates, k)