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
from marketplace_agent.content.examples import load_content_examples
from marketplace_agent.evals.content_few_shot import (
    run_few_shot_comparison,
    serialize_case_result,
)
from marketplace_agent.evals.content_pipeline import (
    load_content_pipeline_products,
)
from marketplace_agent.llm.factory import create_llm_client


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare zero-shot and few-shot content generation."
    )
    parser.add_argument("--zero-shot-version", required=True)
    parser.add_argument("--few-shot-version", required=True)
    parser.add_argument("--input-price-per-million", type=float)
    parser.add_argument("--output-price-per-million", type=float)
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser.parse_args()


def configured_model(settings: Settings) -> str:
    return {
        "groq": settings.groq_model,
        "openrouter": settings.openrouter_model,
        "gigachat": settings.gigachat_model,
    }[settings.llm_provider]


def print_progress(event: dict[str, object]) -> None:
    print(
        f"[{event['version']}] {event['position']}/{event['total']} "
        f"{event['sku']}: {event['event_type']}",
        flush=True,
    )


def main() -> None:
    arguments = parse_arguments()
    manifest_path = (
        PROJECT_ROOT / "data/gold/content_pipeline_manifest.json"
    )
    examples_path = PROJECT_ROOT / "data/gold/content_examples.json"
    products = load_content_pipeline_products(manifest_path)
    examples = load_content_examples(examples_path)

    settings = Settings.from_environment()
    input_price = (
        arguments.input_price_per_million
        if arguments.input_price_per_million is not None
        else settings.input_price_per_million
    )
    output_price = (
        arguments.output_price_per_million
        if arguments.output_price_per_million is not None
        else settings.output_price_per_million
    )
    progress = print_progress if os.getenv("LLM_PROGRESS") == "1" else None

    experiment = run_few_shot_comparison(
        products=products,
        llm_factory=lambda: create_llm_client(settings),
        examples=examples,
        max_attempts=arguments.max_attempts,
        input_price_per_million=input_price,
        output_price_per_million=output_price,
        progress=progress,
    )

    run_id = uuid4().hex
    output_path = (
        PROJECT_ROOT
        / "evals/runs"
        / f"content-few-shot-{run_id}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "provider": settings.llm_provider,
                "model": configured_model(settings),
                "manifest_sha256": hashlib.sha256(
                    manifest_path.read_bytes()
                ).hexdigest(),
                "examples_sha256": hashlib.sha256(
                    examples_path.read_bytes()
                ).hexdigest(),
                "prices": {
                    "input_per_million": input_price,
                    "output_per_million": output_price,
                },
                "max_attempts": arguments.max_attempts,
                "zero_shot_version": arguments.zero_shot_version,
                "few_shot_version": arguments.few_shot_version,
                "zero_shot_results": [
                    serialize_case_result(result)
                    for result in experiment.zero_shot_results
                ],
                "few_shot_results": [
                    serialize_case_result(result)
                    for result in experiment.few_shot_results
                ],
                "comparison": {
                    "zero_shot": experiment.comparison.baseline.__dict__,
                    "few_shot": experiment.comparison.validated.__dict__,
                    "corrected_skus": (
                        experiment.comparison.corrected_skus
                    ),
                    "degraded_skus": (
                        experiment.comparison.degraded_skus
                    ),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Run ID: {run_id}")
    print(
        "Zero-shot success rate: "
        f"{experiment.comparison.baseline.success_rate:.3f}"
    )
    print(
        "Few-shot success rate: "
        f"{experiment.comparison.validated.success_rate:.3f}"
    )
    print(f"Results: {output_path}")


if __name__ == "__main__":
    main()