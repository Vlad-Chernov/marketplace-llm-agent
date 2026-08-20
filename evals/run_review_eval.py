import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marketplace_agent.config import Settings
from marketplace_agent.data_generation.catalog import generate_clean_products
from marketplace_agent.data_generation.reviews import generate_reviews
from marketplace_agent.evals.review_metrics import (
    append_review_evaluation_report,
    evaluate_review_classification,
)
from marketplace_agent.llm.factory import create_llm_client
from marketplace_agent.reviews.analyzer import classify_review
from marketplace_agent.reviews.taxonomy import load_defect_taxonomy


def main() -> None:
    products = generate_clean_products(count=10, seed=7)
    reviews = generate_reviews(products, count=40, seed=42)
    defect_reviews = [
        review for review in reviews if review.defect_label is not None
    ][:5]

    taxonomy = load_defect_taxonomy(
        PROJECT_ROOT / "data" / "taxonomy" / "laptop_defects.yaml"
    )
    client = create_llm_client(Settings.from_environment())

    labels_by_review_id = {
        review.review_id: classify_review(review, taxonomy, client)
        for review in defect_reviews
    }
    evaluation = evaluate_review_classification(
        defect_reviews,
        labels_by_review_id,
    )
    report_path = PROJECT_ROOT / "evals" / "REPORT.md"
    append_review_evaluation_report(evaluation, report_path)

    print(f"Reviews: {len(defect_reviews)}")
    print(f"Overall recall: {evaluation.overall_recall:.3f}")
    print(
        "Mean absolute frequency error: "
        f"{evaluation.mean_absolute_frequency_error:.3f}"
    )
    print(f"Weak defects: {', '.join(evaluation.weak_defects) or 'none'}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()