import json
import sys
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marketplace_agent.evals.retrieval_metrics import (
    RetrievalCase,
    append_retrieval_report,
    evaluate_retriever,
)
from marketplace_agent.retrieval.documents import load_policy_chunks
from marketplace_agent.retrieval.hybrid import HybridRetriever
from marketplace_agent.retrieval.lexical import BM25Retriever
from marketplace_agent.retrieval.vector import (
    SentenceTransformerEmbedder,
    VectorRetriever,
)


def load_cases() -> list[RetrievalCase]:
    cases_path = PROJECT_ROOT / "data" / "gold" / "policy_retrieval_cases.json"
    raw_cases = json.loads(cases_path.read_text(encoding="utf-8"))
    return [RetrievalCase.model_validate(raw_case) for raw_case in raw_cases]


def main() -> None:
    cases = load_cases()
    chunks = load_policy_chunks(PROJECT_ROOT / "data" / "support")
    embedder = SentenceTransformerEmbedder()
    bm25 = BM25Retriever(chunks)
    vector = VectorRetriever.open(
        PROJECT_ROOT / "data" / "vector_store",
        embedder,
    )
    hybrid = HybridRetriever(bm25, vector)

    evaluations = {
        "bm25": evaluate_retriever(bm25, cases, k=5),
        "vector": evaluate_retriever(vector, cases, k=5),
        "hybrid": evaluate_retriever(hybrid, cases, k=5),
    }

    results_path = (
        PROJECT_ROOT / "evals" / "runs" / f"retrieval-{uuid4().hex}.json"
    )
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(
            {
                name: evaluation.model_dump()
                for name, evaluation in evaluations.items()
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    append_retrieval_report(
        evaluations,
        PROJECT_ROOT / "evals" / "REPORT.md",
        k=5,
    )

    for name, evaluation in evaluations.items():
        print(f"{name} Recall@5: {evaluation.recall_at_k:.3f}")
    print(f"Results: {results_path}")


if __name__ == "__main__":
    main()