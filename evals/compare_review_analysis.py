import argparse
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marketplace_agent.config import Settings
from marketplace_agent.data_generation.catalog import (
    generate_clean_products,
)
from marketplace_agent.data_generation.reviews import generate_reviews
from marketplace_agent.evals.review_clustering import (
    run_review_clustering_experiment,
    serialize_review_clustering_result,
)
from marketplace_agent.llm.factory import create_llm_client
from marketplace_agent.retrieval.vector import (
    MODEL_NAME,
    SentenceTransformerEmbedder,
)
from marketplace_agent.reviews.taxonomy import load_defect_taxonomy

CATALOG_SEED = 7
REVIEW_SEED = 42
REVIEW_COUNT = 200


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare review taxonomy classification "
            "and local embedding clustering."
        )
    )
    parser.add_argument(
        "--review-count",
        type=int,
        default=REVIEW_COUNT,
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.82,
    )
    return parser.parse_args()


def print_progress(event: dict[str, object]) -> None:
    print(
        f"[taxonomy] {event['position']}/{event['total']} "
        f"{event['review_id']}: {event['event_type']}",
        flush=True,
    )


def dataset_sha256(reviews: list[object]) -> str:
    payload = json.dumps(
        [
            review.model_dump(mode="json")
            for review in reviews
        ],
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def main() -> None:
    arguments = parse_arguments()

    if arguments.review_count != REVIEW_COUNT:
        raise ValueError("review-count must be 200 for E-005.")

    products = generate_clean_products(
        count=10,
        seed=CATALOG_SEED,
    )
    reviews = generate_reviews(
        products,
        count=REVIEW_COUNT,
        seed=REVIEW_SEED,
    )
    taxonomy = load_defect_taxonomy(
        PROJECT_ROOT / "data/taxonomy/laptop_defects.yaml"
    )
    progress = print_progress if os.getenv("LLM_PROGRESS") == "1" else None

    result = run_review_clustering_experiment(
        reviews=reviews,
        taxonomy=taxonomy,
        llm=create_llm_client(Settings.from_environment()),
        embedder=SentenceTransformerEmbedder(),
        similarity_threshold=arguments.similarity_threshold,
        progress=progress,
    )

    run_id = uuid4().hex
    output_path = (
        PROJECT_ROOT
        / "evals/runs"
        / f"review-clustering-{run_id}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "catalog_seed": CATALOG_SEED,
                "review_seed": REVIEW_SEED,
                "review_count": REVIEW_COUNT,
                "dataset_sha256": dataset_sha256(reviews),
                "embedding_model": MODEL_NAME,
                "similarity_threshold": (
                    arguments.similarity_threshold
                ),
                "result": serialize_review_clustering_result(
                    result
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Cleaned reviews: {result.cleaned_review_count}")
    print(
        "Taxonomy recall: "
        f"{result.taxonomy_evaluation.overall_recall:.3f}"
    )
    print(
        "Cluster defect recall: "
        f"{result.cluster_evaluation.defect_recall:.3f}"
    )
    print(f"Taxonomy latency: {result.taxonomy_latency_ms} ms")
    print(f"Clustering latency: {result.clustering_latency_ms} ms")
    print(f"Results: {output_path}")


if __name__ == "__main__":
    main()