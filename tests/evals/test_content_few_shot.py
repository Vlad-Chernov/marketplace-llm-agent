from collections.abc import Sequence
from decimal import Decimal

from marketplace_agent.content.examples import ContentExample
from marketplace_agent.domain.models import (
    GeneratedContent,
    PipelineResult,
    Product,
)
from marketplace_agent.evals.content_few_shot import (
    run_few_shot_comparison,
    serialize_case_result,
)
from marketplace_agent.evals.content_pipeline import (
    ContentPipelineCaseResult,
)
from marketplace_agent.llm.base import FakeLLMClient


def make_product(sku: str) -> Product:
    return Product(
        sku=sku,
        category="laptops",
        brand="Lenovo",
        model="IdeaPad",
        price=Decimal(75000),
        sales_count=10,
        supplier_description="Ноутбук Lenovo.",
        attributes={},
    )


def make_example() -> ContentExample:
    return ContentExample(
        example_id="EXAMPLE-001",
        category="laptops",
        confirmed_attributes={"ram_gb": "16"},
        content=GeneratedContent(
            title="Ноутбук для работы",
            bullets=["16 GB RAM"],
            description="Карточка-пример.",
            keywords=["ноутбук"],
            used_attributes={"ram_gb": "16"},
        ),
    )


def test_runs_versions_with_separate_clients() -> None:
    created_clients: list[FakeLLMClient] = []
    seen_example_ids: list[tuple[str, ...]] = []

    def llm_factory() -> FakeLLMClient:
        client = FakeLLMClient()
        created_clients.append(client)
        return client

    def completed_pipeline(
        product: Product,
        _: FakeLLMClient,
        __: int,
        *,
        examples: Sequence[ContentExample] = (),
    ) -> PipelineResult:
        seen_example_ids.append(
            tuple(example.example_id for example in examples)
        )
        return PipelineResult(
            sku=product.sku,
            attempts=1,
            status="completed",
            selected_example_ids=[
                example.example_id
                for example in examples
            ],
        )

    result = run_few_shot_comparison(
        products=[
            make_product("LAP-0001"),
            make_product("LAP-0002"),
        ],
        llm_factory=llm_factory,
        examples=[make_example()],
        max_attempts=3,
        input_price_per_million=0.0,
        output_price_per_million=0.0,
        pipeline=completed_pipeline,
    )

    assert len(created_clients) == 4
    assert seen_example_ids == [
        (),
        (),
        ("EXAMPLE-001",),
        ("EXAMPLE-001",),
    ]
    assert [
        case.selected_example_ids
        for case in result.zero_shot_results
    ] == [[], []]
    assert [
        case.selected_example_ids
        for case in result.few_shot_results
    ] == [["EXAMPLE-001"], ["EXAMPLE-001"]]
    assert result.comparison.baseline.success_rate == 1.0
    assert result.comparison.validated.success_rate == 1.0

def test_serializes_case_result_without_hidden_attributes() -> None:
    case = ContentPipelineCaseResult(
        sku="LAP-0001",
        status="completed",
        attempts=1,
        true_attributes={"supplier_secret": "не публиковать"},
        extracted_attributes={"ram_gb": "16"},
        used_attributes={"ram_gb": "16"},
        violations=[],
        latency_ms=120,
        cost_usd=0.01,
        selected_example_ids=["EXAMPLE-001"],
    )

    payload = serialize_case_result(case)

    assert payload["sku"] == "LAP-0001"
    assert payload["selected_example_ids"] == ["EXAMPLE-001"]
    assert "true_attributes" not in payload
    assert "extracted_attributes" not in payload
    assert "used_attributes" not in payload