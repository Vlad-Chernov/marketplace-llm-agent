import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from decimal import Decimal
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from marketplace_agent.config import Settings
from marketplace_agent.content.pipeline import run_content_pipeline
from marketplace_agent.domain.models import PipelineResult, Product
from marketplace_agent.evals.llm_meter import MeteredLLMClient
from marketplace_agent.llm.base import Message
from marketplace_agent.llm.factory import create_llm_client
from marketplace_agent.llm.providers import LLMProviderError
from marketplace_agent.llm.telemetry import (
    TracingLLMClient,
    create_trace_writer,
    trace_event,
    trace_run,
)
from marketplace_agent.retrieval.documents import load_policy_chunks
from marketplace_agent.retrieval.lexical import BM25Retriever
from marketplace_agent.storage.repositories import (
    OrderRepository,
    ProductRepository,
)
from marketplace_agent.support.agent import AgentAnswer, SupportAgent
from marketplace_agent.support.registry import ToolRegistry
from marketplace_agent.support.tools import (
    GetOrderTool,
    GetProductTool,
    SearchPolicyTool,
)

app = FastAPI(
    title="Marketplace Agent API",
    version="0.1.0",
    description="HTTP API for marketplace agent workflows.",
)
_job_executor = ThreadPoolExecutor(max_workers=4)
_jobs: dict[str, dict[str, object]] = {}
_jobs_lock = Lock()


@app.exception_handler(LLMProviderError)
async def handle_llm_provider_error(
    _request: Request,
    error: LLMProviderError,
) -> JSONResponse:
    """Expose a safe provider failure reason to API clients."""

    return JSONResponse(status_code=502, content={"detail": str(error)})


class ContentDemoRequest(BaseModel):
    supplier_description: str = Field(min_length=1)
    brand: str = Field(min_length=1)
    model: str = Field(min_length=1)
    category: str = "laptops"
    sku: str = "DEMO-001"
    price: Decimal = Field(default=Decimal(1), gt=0)
    sales_count: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=1, ge=1, le=3)


class SupportDemoRequest(BaseModel):
    """Store one safe support-chat request."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1)
    session_id: str = Field(default="demo-session", min_length=1)
    history: list[Message] = Field(default_factory=list)


class SupportDemoResponse(BaseModel):
    """Return a support answer and observable request timing."""

    model_config = ConfigDict(extra="forbid")

    answer: AgentAnswer
    latency_ms: int = Field(ge=0)


class DemoJobResponse(BaseModel):
    """Expose the status of a background demo operation."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    kind: str
    status: str
    result: dict[str, object] | None = None
    error: str | None = None


@app.get("/health")
def health() -> dict[str, str]:
    """Return a lightweight liveness response."""

    return {"status": "ok"}


@app.get("/demo/reviews/report")
def review_report() -> dict[str, object]:
    """Return the latest safe review-evaluation summary."""

    return _load_latest_review_report()


@app.post("/demo/support", response_model=SupportDemoResponse)
def support_demo(request: SupportDemoRequest) -> SupportDemoResponse:
    """Answer one support question through the guarded support agent."""

    return _run_with_trace(
        "support",
        lambda: _run_support_request_response(request),
    )


@app.post("/demo/support/jobs", status_code=202)
def start_support_job(request: SupportDemoRequest) -> DemoJobResponse:
    """Start support processing without blocking the browser session."""

    return _submit_job(
        "support",
        lambda: _run_support_request_response(request),
    )


@app.post("/demo/content", response_model=PipelineResult)
def generate_demo_content(request: ContentDemoRequest) -> PipelineResult:
    """Generate one product card through the production content pipeline."""

    return _run_with_trace(
        "content",
        lambda: _run_content_request(request),
    )


@app.post("/demo/content/jobs", status_code=202)
def start_content_job(request: ContentDemoRequest) -> DemoJobResponse:
    """Start content processing without blocking the browser session."""

    return _submit_job(
        "content",
        lambda: _run_content_request(request),
    )


@app.get("/demo/jobs/{job_id}", response_model=DemoJobResponse)
def get_demo_job(job_id: str) -> DemoJobResponse:
    """Return a background job status and its completed safe result."""

    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return DemoJobResponse(
                job_id=job_id,
                kind="unknown",
                status="not_found",
                error="Задача не найдена.",
            )
        return DemoJobResponse(**job)


def _run_content_request(request: ContentDemoRequest) -> PipelineResult:
    settings = Settings.from_environment()
    gigachat_settings = replace(settings, llm_provider="gigachat")
    if not gigachat_settings.gigachat_authorization_key:
        raise ValueError("GIGACHAT_AUTHORIZATION_KEY is required.")

    client = MeteredLLMClient(
        TracingLLMClient(create_llm_client(gigachat_settings)),
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


def _submit_job(
    kind: str,
    work: Any,
) -> DemoJobResponse:
    job_id = uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {
            "job_id": job_id,
            "kind": kind,
            "status": "queued",
            "result": None,
            "error": None,
        }
    _job_executor.submit(_execute_job, job_id, work)
    return DemoJobResponse(**_jobs[job_id])


def _execute_job(job_id: str, work: Any) -> None:
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
        kind = str(_jobs[job_id]["kind"])
    run_id = f"{kind}-{job_id}"
    writer = create_trace_writer(
        _project_root() / "data" / "traces",
        run_id,
    )
    try:
        with trace_run(writer, run_id):
            trace_event("job_started", {"kind": kind})
            result = work()
            serialized = result.model_dump(mode="json")
            trace_event(
                "job_completed",
                {"kind": kind, "status": "completed"},
            )
    except Exception as error:  # noqa: BLE001
        with _jobs_lock:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(error)
        return
    with _jobs_lock:
        _jobs[job_id]["status"] = "completed"
        _jobs[job_id]["result"] = serialized


def _load_latest_review_report() -> dict[str, object]:
    run_files = sorted(
        (_project_root() / "evals" / "runs").glob(
            "review-clustering-*.json"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not run_files:
        return {
            "status": "unavailable",
            "message": "Отчёт по отзывам ещё не запускался.",
        }

    payload: dict[str, Any] = json.loads(
        run_files[0].read_text(encoding="utf-8")
    )
    result = payload["result"]
    return {
        "status": "available",
        "run_id": payload["run_id"],
        "created_at": payload["created_at"],
        "review_count": payload["review_count"],
        "result": {
            "cleaned_review_count": result["cleaned_review_count"],
            "taxonomy_metrics": result["taxonomy_metrics"],
            "cluster_metrics": result["cluster_metrics"],
            "taxonomy_error_types": result["taxonomy_error_types"],
        },
    }


def _run_support_request(request: SupportDemoRequest) -> AgentAnswer:
    settings = Settings.from_environment()
    gigachat_settings = replace(settings, llm_provider="gigachat")
    if not gigachat_settings.gigachat_authorization_key:
        raise ValueError("GIGACHAT_AUTHORIZATION_KEY is required.")

    llm = MeteredLLMClient(
        TracingLLMClient(create_llm_client(gigachat_settings)),
        input_price_per_million=gigachat_settings.input_price_per_million,
        output_price_per_million=gigachat_settings.output_price_per_million,
    )
    root = _project_root()
    database_path = root / "data" / "synthetic" / "marketplace.db"
    retriever = BM25Retriever(
        load_policy_chunks(root / "data" / "support")
    )
    registry = ToolRegistry(
        [
            GetProductTool(ProductRepository(database_path)),
            GetOrderTool(OrderRepository(database_path)),
            SearchPolicyTool(retriever),
        ]
    )
    return SupportAgent(registry, llm).run(
        message=request.message,
        session_id=request.session_id,
        history=request.history,
    )


def _run_support_request_response(
    request: SupportDemoRequest,
) -> SupportDemoResponse:
    started_at = perf_counter()
    answer = _run_support_request(request)
    return SupportDemoResponse(
        answer=answer,
        latency_ms=round((perf_counter() - started_at) * 1000),
    )


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _run_with_trace(kind: str, work: Callable[[], Any]) -> Any:
    run_id = f"{kind}-{uuid4().hex}"
    writer = create_trace_writer(
        _project_root() / "data" / "traces",
        run_id,
    )
    with trace_run(writer, run_id):
        return work()
