import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from marketplace_agent.config import Settings
from marketplace_agent.data_generation.specs import load_attribute_specs
from marketplace_agent.evals.llm_meter import MeteredLLMClient
from marketplace_agent.evals.mvp_executor import MvpCaseExecutor
from marketplace_agent.evals.mvp_final import (
    load_golden_cases,
    run_mvp_final,
)
from marketplace_agent.evals.recording_registry import (
    RecordingToolRegistry,
)
from marketplace_agent.evals.runner import save_run
from marketplace_agent.llm.cache import CachedLLMClient
from marketplace_agent.llm.factory import create_llm_client
from marketplace_agent.retrieval.documents import load_policy_chunks
from marketplace_agent.retrieval.hybrid import HybridRetriever
from marketplace_agent.retrieval.lexical import BM25Retriever
from marketplace_agent.retrieval.vector import (
    SentenceTransformerEmbedder,
    VectorRetriever,
)
from marketplace_agent.reviews.taxonomy import load_defect_taxonomy
from marketplace_agent.storage.repositories import (
    OrderRepository,
    ProductRepository,
)
from marketplace_agent.support.registry import ToolRegistry
from marketplace_agent.support.tools import (
    GetOrderTool,
    GetProductTool,
    SearchPolicyTool,
)


def main() -> None:
    """Run all MVP golden cases through the real application modules."""

    settings = Settings.from_environment()
    cached_llm = CachedLLMClient(
        create_llm_client(settings),
        cache_path=(
            PROJECT_ROOT / "data" / "cache" / "mvp-final-llm.json"
        ),
    )
    meter = MeteredLLMClient(
        cached_llm,
        input_price_per_million=settings.input_price_per_million,
        output_price_per_million=settings.output_price_per_million,
    )

    chunks = load_policy_chunks(PROJECT_ROOT / "data" / "support")
    retriever = HybridRetriever(
        lexical=BM25Retriever(chunks),
        vector=VectorRetriever.open(
            PROJECT_ROOT / "data" / "vector_store",
            SentenceTransformerEmbedder(),
        ),
    )

    database_path = (
        PROJECT_ROOT / "data" / "synthetic" / "marketplace.db"
    )
    product_repository = ProductRepository(database_path)
    order_repository = OrderRepository(database_path)
    registry = ToolRegistry(
        [
            GetProductTool(product_repository),
            GetOrderTool(order_repository),
            SearchPolicyTool(retriever),
        ]
    )
    executor = MvpCaseExecutor(
        llm=meter,
        attribute_specs=load_attribute_specs(
            PROJECT_ROOT / "data" / "specs" / "laptops.yaml"
        ),
        defect_taxonomy=load_defect_taxonomy(
            PROJECT_ROOT / "data" / "taxonomy" / "laptop_defects.yaml"
        ),
        support_registry=RecordingToolRegistry(registry),
    )

    cases = load_golden_cases(
        PROJECT_ROOT / "data" / "gold" / "mvp_cases.json"
    )
    print(
        f"Cache: {cached_llm.cache_hits} hits, "
        f"{cached_llm.cache_misses} misses"
    )
    run = run_mvp_final(cases, executor.execute)
    output_path = save_run(
        run,
        PROJECT_ROOT / "evals" / "results",
    )

    errors = sum(result.error is not None for result in run.results)
    print(
        f"Cache: {cached_llm.cache_hits} hits, "
        f"{cached_llm.cache_misses} misses"
    )
    print(f"Saved evaluation: {output_path}")
    print(f"Cases: {len(run.results)}, errors: {errors}")


if __name__ == "__main__":
    main()