from dataclasses import replace
from decimal import Decimal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from marketplace_agent.config import Settings
from marketplace_agent.content.pipeline import run_content_pipeline
from marketplace_agent.domain.models import PipelineResult, Product
from marketplace_agent.evals.llm_meter import MeteredLLMClient
from marketplace_agent.llm.factory import create_llm_client

app = FastAPI(
    title="Marketplace Agent API",
    version="0.1.0",
    description="HTTP API for marketplace agent workflows.",
)


class ContentDemoRequest(BaseModel):
    supplier_description: str = Field(min_length=1)
    brand: str = Field(min_length=1)
    model: str = Field(min_length=1)
    category: str = "laptops"
    sku: str = "DEMO-001"
    price: Decimal = Field(default=Decimal(1), gt=0)
    sales_count: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=1, ge=1, le=3)


@app.get("/health")
def health() -> dict[str, str]:
    """Return a lightweight liveness response."""

    return {"status": "ok"}


@app.post("/demo/content", response_model=PipelineResult)
def generate_demo_content(request: ContentDemoRequest) -> PipelineResult:
    """Generate one product card through the production content pipeline."""

    settings = Settings.from_environment()
    gigachat_settings = replace(settings, llm_provider="gigachat")
    if not gigachat_settings.gigachat_authorization_key:
        raise ValueError("GIGACHAT_AUTHORIZATION_KEY is required.")

    client = MeteredLLMClient(
        create_llm_client(gigachat_settings),
        input_price_per_million=gigachat_settings.input_price_per_million,
        output_price_per_million=gigachat_settings.output_price_per_million,
    )
    product = Product(
        sku=request.sku,
        category=request.category,
        brand=request.brand,
        model=request.model,
        price=request.price,
        sales_count=request.sales_count,
        supplier_description=request.supplier_description,
    )
    return run_content_pipeline(product, client, max_attempts=request.max_attempts)
