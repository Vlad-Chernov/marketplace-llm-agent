import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marketplace_agent.config import Settings
from marketplace_agent.data_generation.specs import load_attribute_specs
from marketplace_agent.evals.mvp_executor import MvpCaseExecutor
from marketplace_agent.evals.mvp_final import load_golden_cases
from marketplace_agent.evals.provider_comparison import (
    ProviderSpec,
    run_provider_comparison,
    serialize_provider_comparison,
)
from marketplace_agent.evals.recording_registry import RecordingToolRegistry
from marketplace_agent.llm.factory import create_llm_client
from marketplace_agent.retrieval.documents import load_policy_chunks
from marketplace_agent.retrieval.hybrid import HybridRetriever
from marketplace_agent.retrieval.lexical import BM25Retriever
from marketplace_agent.retrieval.vector import (
    SentenceTransformerEmbedder,
    VectorRetriever,
)
from marketplace_agent.reviews.taxonomy import load_defect_taxonomy
from marketplace_agent.storage.repositories import OrderRepository, ProductRepository
from marketplace_agent.support.registry import ToolRegistry
from marketplace_agent.support.tools import (
    GetOrderTool,
    GetProductTool,
    SearchPolicyTool,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare GigaChat and Groq on the MVP golden cases."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=PROJECT_ROOT / "data" / "gold" / "mvp_cases.json",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=25,
        help="Number of cases to run from the beginning of the dataset (default: 25).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    settings = Settings.from_environment()
    _require_keys(settings)
    if args.limit < 1:
        raise SystemExit("--limit must be at least 1")
    cases = load_golden_cases(args.cases)
    cases = cases[: args.limit]
    providers = _build_providers(settings)

    result = run_provider_comparison(
        cases,
        providers,
        progress_callback=_print_progress if os.getenv("LLM_PROGRESS") == "1" else None,
    )
    output_dir = PROJECT_ROOT / "evals" / "runs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"provider-comparison-{uuid4().hex}.json"
    output_path.write_text(
        json.dumps(serialize_provider_comparison(result), ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print(f"Results: {output_path}")
    for run in result.runs:
        print(
            f"{run.provider}: errors={run.error_count}, "
            f"p50={run.latency_p50_ms} ms, p95={run.latency_p95_ms} ms, "
            f"error_types={dict(run.error_types)}"
        )


def _build_providers(settings: Settings) -> list[ProviderSpec]:
    shared = _build_executor_dependencies()
    specs: list[ProviderSpec] = []
    for provider in ("gigachat", "groq"):
        provider_settings = replace(settings, llm_provider=provider)
        specs.append(
            ProviderSpec(
                name=provider,
                model=(
                    provider_settings.gigachat_model
                    if provider == "gigachat"
                    else provider_settings.groq_model
                ),
                api_key=(
                    provider_settings.gigachat_authorization_key
                    if provider == "gigachat"
                    else provider_settings.groq_api_key
                ),
                client_factory=lambda current=provider_settings: create_llm_client(current),
                executor_factory=lambda meter, deps=shared: MvpCaseExecutor(
                    llm=meter,
                    attribute_specs=deps[0],
                    defect_taxonomy=deps[1],
                    support_registry=deps[2],
                ),
                input_price_per_million=provider_settings.input_price_per_million,
                output_price_per_million=provider_settings.output_price_per_million,
            )
        )
    return specs


def _build_executor_dependencies() -> tuple[object, object, RecordingToolRegistry]:
    chunks = load_policy_chunks(PROJECT_ROOT / "data" / "support")
    retriever = HybridRetriever(
        lexical=BM25Retriever(chunks),
        vector=VectorRetriever.open(
            PROJECT_ROOT / "data" / "vector_store",
            SentenceTransformerEmbedder(),
        ),
    )
    database_path = PROJECT_ROOT / "data" / "synthetic" / "marketplace.db"
    registry = ToolRegistry(
        [
            GetProductTool(ProductRepository(database_path)),
            GetOrderTool(OrderRepository(database_path)),
            SearchPolicyTool(retriever),
        ]
    )
    return (
        load_attribute_specs(PROJECT_ROOT / "data" / "specs" / "laptops.yaml"),
        load_defect_taxonomy(PROJECT_ROOT / "data" / "taxonomy" / "laptop_defects.yaml"),
        RecordingToolRegistry(registry),
    )


def _require_keys(settings: Settings) -> None:
    missing = []
    if not settings.gigachat_authorization_key:
        missing.append("GIGACHAT_AUTHORIZATION_KEY")
    if not settings.groq_api_key:
        missing.append("GROQ_API_KEY")
    if missing:
        raise SystemExit(f"Missing required environment variable(s): {', '.join(missing)}")


def _print_progress(provider: str, case_id: str, current: int, total: int, status: str) -> None:
    print(f"[{provider}] {current}/{total} {case_id}: {status}", flush=True)


if __name__ == "__main__":
    main()
