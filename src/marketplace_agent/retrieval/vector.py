from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.errors import NotFoundError
from sentence_transformers import SentenceTransformer

from marketplace_agent.retrieval.documents import PolicyChunk
from marketplace_agent.retrieval.lexical import SearchResult

COLLECTION_NAME = "policy_chunks"
MODEL_NAME = "intfloat/multilingual-e5-small"


class Embedder(Protocol):
    """Encode documents and queries into normalized vectors."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_queries(self, texts: list[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Create E5 embeddings for documents and queries."""

    def __init__(self, model_name: str = MODEL_NAME) -> None:
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode(["passage: " + text for text in texts])

    def embed_queries(self, texts: list[str]) -> list[list[float]]:
        return self._encode(["query: " + text for text in texts])

    def _encode(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
        )
        return embeddings.tolist()


class VectorRetriever:
    """Search a persistent ChromaDB collection."""

    def __init__(self, collection: Collection, embedder: Embedder) -> None:
        self._collection = collection
        self._embedder = embedder

    @property
    def count(self) -> int:
        """Return the number of indexed chunks."""

        return self._collection.count()

    @classmethod
    def open(
        cls,
        path: Path | str,
        embedder: Embedder,
    ) -> "VectorRetriever":
        """Open the persistent policy collection."""

        client = chromadb.PersistentClient(path=str(path))
        collection = client.get_collection(
            COLLECTION_NAME,
            embedding_function=None,
        )
        return cls(collection, embedder)

    def search(
        self,
        query: str,
        k: int,
        filters: Mapping[str, str] | None = None,
    ) -> list[SearchResult]:
        """Return the nearest policy chunks for a query."""

        if not query.strip() or k <= 0 or self.count == 0:
            return []

        where = _build_where(filters)
        if filters is not None and where is None:
            return []

        result = self._collection.query(
            query_embeddings=self._embedder.embed_queries([query]),
            n_results=min(k, self.count),
            where=where,
            include=["metadatas", "distances"],
        )
        chunk_ids = result["ids"][0]
        metadatas = result["metadatas"][0]
        distances = result["distances"][0]

        return [
            SearchResult(
                chunk=_chunk_from_metadata(chunk_id, metadata),
                score=1.0 - float(distance),
                rank=rank,
            )
            for rank, (chunk_id, metadata, distance) in enumerate(
                zip(chunk_ids, metadatas, distances, strict=True),
                start=1,
            )
        ]


def build_vector_index(
    chunks: Sequence[PolicyChunk],
    path: Path | str,
    embedder: Embedder,
) -> VectorRetriever:
    """Rebuild the persistent policy collection from source chunks."""

    client = chromadb.PersistentClient(path=str(path))

    try:
        client.delete_collection(COLLECTION_NAME)
    except NotFoundError:
        pass

    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=None,
    )
    chunk_list = list(chunks)

    if chunk_list:
        collection.add(
            ids=[chunk.chunk_id for chunk in chunk_list],
            embeddings=embedder.embed_documents(
                [chunk.text for chunk in chunk_list]
            ),
            metadatas=[
                {
                    "document_id": chunk.document_id,
                    "policy_type": chunk.policy_type,
                    "heading": chunk.heading,
                    "text": chunk.text,
                }
                for chunk in chunk_list
            ],
        )

    return VectorRetriever(collection, embedder)


def _build_where(
    filters: Mapping[str, str] | None,
) -> dict[str, object] | None:
    if filters is None:
        return None

    if not filters or any(
        key not in {"document_id", "policy_type"} for key in filters
    ):
        return None

    if len(filters) == 1:
        return dict(filters)

    return {
        "$and": [{key: value} for key, value in filters.items()],
    }


def _chunk_from_metadata(
    chunk_id: str,
    metadata: dict[str, str],
) -> PolicyChunk:
    return PolicyChunk(
        chunk_id=chunk_id,
        document_id=metadata["document_id"],
        policy_type=metadata["policy_type"],
        heading=metadata["heading"],
        text=metadata["text"],
    )