import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marketplace_agent.config import Settings
from marketplace_agent.demo import format_demo_result, run_demo
from marketplace_agent.llm.factory import create_llm_client
from marketplace_agent.llm.progress import ProgressLLMClient
from marketplace_agent.retrieval.documents import load_policy_chunks
from marketplace_agent.retrieval.hybrid import HybridRetriever
from marketplace_agent.retrieval.lexical import BM25Retriever
from marketplace_agent.retrieval.vector import (
    SentenceTransformerEmbedder,
    VectorRetriever,
)


def main() -> None:
    """Run the real end-to-end MVP demonstration."""

    chunks = load_policy_chunks(PROJECT_ROOT / "data" / "support")
    retriever = HybridRetriever(
        lexical=BM25Retriever(chunks),
        vector=VectorRetriever.open(
            PROJECT_ROOT / "data" / "vector_store",
            SentenceTransformerEmbedder(),
        ),
    )
    llm = create_llm_client(Settings.from_environment())
    if os.getenv("LLM_PROGRESS") == "1":
        llm = ProgressLLMClient(llm)

    result = run_demo(
        database_path=PROJECT_ROOT / "data" / "synthetic" / "marketplace.db",
        retriever=retriever,
        llm=llm,
    )
    print(format_demo_result(result))


if __name__ == "__main__":
    main()
