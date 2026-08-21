import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marketplace_agent.retrieval.documents import load_policy_chunks
from marketplace_agent.retrieval.vector import (
    SentenceTransformerEmbedder,
    build_vector_index,
)


def main() -> None:
    chunks = load_policy_chunks(PROJECT_ROOT / "data" / "support")
    retriever = build_vector_index(
        chunks,
        PROJECT_ROOT / "data" / "vector_store",
        SentenceTransformerEmbedder(),
    )
    print(f"Indexed chunks: {retriever.count}")


if __name__ == "__main__":
    main()