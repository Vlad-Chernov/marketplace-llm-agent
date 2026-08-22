from pathlib import Path

import pytest

from marketplace_agent.data_generation.specs import load_attribute_specs
from marketplace_agent.evals.llm_meter import MeteredLLMClient
from marketplace_agent.evals.models import GoldenCase
from marketplace_agent.evals.mvp_executor import MvpCaseExecutor
from marketplace_agent.evals.recording_registry import (
    RecordingToolRegistry,
)
from marketplace_agent.llm.base import FakeLLMClient, LLMResponse
from marketplace_agent.reviews.taxonomy import load_defect_taxonomy
from marketplace_agent.support.tools import ToolResult


class SupportPolicyRegistry:
    def schemas(self) -> list[dict[str, object]]:
        return [
            {
                "type": "function",
                "function": {"name": "search_policy"},
            }
        ]

    def run(
        self,
        name: str,
        arguments: dict[str, object],
    ) -> ToolResult:
        assert name == "search_policy"
        assert arguments == {"query": "Возврат товара"}

        return ToolResult(
            ok=True,
            data=[
                {
                    "chunk_id": "returns-01",
                    "text": "Возврат возможен в течение 14 дней.",
                }
            ],
            citations=["returns-01"],
        )

def test_executes_attribute_extraction_case() -> None:
    raw_response = (
        '{"attributes":{'
        '"ram_gb":{"value":"16","confidence":0.95},'
        '"storage_gb":{"value":"512","confidence":0.95}'
        '},'
        '"unsupported_facts":[]}'
    )
    meter = MeteredLLMClient(
        FakeLLMClient(
            [
                LLMResponse(
                    content=raw_response,
                    model="fake-model",
                    prompt_tokens=100,
                    completion_tokens=20,
                )
            ]
        ),
        input_price_per_million=1.0,
        output_price_per_million=2.0,
    )
    case = GoldenCase(
        id="gold-executor-001",
        question="Извлеки характеристики.",
        type="attribute_extraction",
        input={
            "supplier_description": "Ноутбук: ОЗУ 16 ГБ, SSD 512 ГБ.",
        },
        expected_answer={"ram_gb": "16", "storage_gb": "512"},
        origin="manual",
    )
    executor = MvpCaseExecutor(
        llm=meter,
        attribute_specs=load_attribute_specs(
            Path("data/specs/laptops.yaml")
        ),
    )

    prediction = executor.execute(case)

    assert prediction.prediction == {
        "ram_gb": "16",
        "storage_gb": "512",
    }
    assert prediction.raw_response == raw_response
    assert prediction.model == "fake-model"
    assert prediction.prompt_tokens == 100
    assert prediction.completion_tokens == 20
    assert prediction.cost_usd == pytest.approx(0.00014)

def test_executes_deterministic_validation_case() -> None:
    raw_response = '{"violations":[]}'
    meter = MeteredLLMClient(
        FakeLLMClient(
            [
                LLMResponse(
                    content=raw_response,
                    model="fake-model",
                    prompt_tokens=30,
                    completion_tokens=5,
                )
            ]
        ),
        input_price_per_million=1.0,
        output_price_per_million=2.0,
    )
    case = GoldenCase(
        id="gold-executor-002",
        question="Проверь карточку.",
        type="validation",
        input={
            "content": {
                "title": "Ноутбук для работы дома",
                "bullets": [],
                "description": (
                    "Закажите на https://example.com "
                    "или звоните +7 999 123-45-67"
                ),
                "keywords": [],
                "used_attributes": {},
            }
        },
        expected_answer={
            "violations": [
                "contact-information",
                "external-url",
            ]
        },
        origin="manual",
    )
    executor = MvpCaseExecutor(
        llm=meter,
        attribute_specs=load_attribute_specs(
            Path("data/specs/laptops.yaml")
        ),
    )

    prediction = executor.execute(case)

    assert prediction.prediction == {
        "violations": [
            "contact-information",
            "external-url",
        ]
    }
    assert prediction.raw_response == raw_response
    assert prediction.model == "fake-model"
    assert prediction.prompt_tokens == 30
    assert prediction.completion_tokens == 5
    assert prediction.cost_usd == pytest.approx(0.00004)

def test_executes_semantic_validation_case() -> None:
    raw_response = (
        '{"violations":['
        '{"rule_id":"unverifiable-superlative",'
        '"field":"title",'
        '"evidence":"СУПЕР"}'
        ']}'
    )
    meter = MeteredLLMClient(
        FakeLLMClient(
            [
                LLMResponse(
                    content=raw_response,
                    model="fake-model",
                    prompt_tokens=50,
                    completion_tokens=10,
                )
                
            ]
        ),
        input_price_per_million=1.0,
        output_price_per_million=2.0,
    )
    case = GoldenCase(
        id="gold-executor-003",
        question="Проверь карточку.",
        type="validation",
        input={
            "content": {
                "title": "СУПЕР НОУТБУК LENOVO",
                "bullets": [],
                "description": "",
                "keywords": [],
                "used_attributes": {},
            }
        },
        expected_answer={
            "violations": [
                "uppercase-ratio",
                "unverifiable-superlative",
            ]
        },
        origin="manual",
    )
    executor = MvpCaseExecutor(
        llm=meter,
        attribute_specs=load_attribute_specs(
            Path("data/specs/laptops.yaml")
        ),
    )

    prediction = executor.execute(case)

    assert prediction.prediction == {
        "violations": [
            "uppercase-ratio",
            "unverifiable-superlative",
        ]
    }
    assert prediction.raw_response == raw_response
    assert prediction.model == "fake-model"
    assert prediction.prompt_tokens == 50
    assert prediction.completion_tokens == 10
    assert prediction.cost_usd == pytest.approx(0.00007)

def test_executes_review_analysis_case() -> None:
    raw_response = (
        '{"defect_id":"battery_drain",'
        '"kind":"product_defect",'
        '"severity":"high",'
        '"confidence":0.95}'
    )
    meter = MeteredLLMClient(
        FakeLLMClient(
            [
                LLMResponse(
                    content=raw_response,
                    model="fake-model",
                    prompt_tokens=40,
                    completion_tokens=10,
                )
            ]
        ),
        input_price_per_million=1.0,
        output_price_per_million=2.0,
    )
    case = GoldenCase(
        id="gold-executor-004",
        question="Классифицируй отзыв.",
        type="review_analysis",
        input={
            "review": {
                "review_id": "REV-900023",
                "sku": "LAP-0001",
                "rating": 1,
                "text": (
                    "После недели использования аккумулятор "
                    "быстро разряжается."
                ),
                "created_at": "2026-08-20",
                "helpful_count": 0,
            }
        },
        expected_answer={"defect_label": "battery_drain"},
        origin="manual",
    )
    executor = MvpCaseExecutor(
        llm=meter,
        attribute_specs=load_attribute_specs(
            Path("data/specs/laptops.yaml")
        ),
        defect_taxonomy=load_defect_taxonomy(
            Path("data/taxonomy/laptop_defects.yaml")
        ),
    )

    prediction = executor.execute(case)

    assert prediction.prediction == {
        "defect_label": "battery_drain"
    }
    assert prediction.raw_response == raw_response
    assert prediction.model == "fake-model"
    assert prediction.prompt_tokens == 40
    assert prediction.completion_tokens == 10
    assert prediction.cost_usd == pytest.approx(0.00006)

def test_executes_support_case() -> None:
    final_response = (
        '{"kind":"final","status":"answered",'
        '"text":"Вернуть товар можно в течение 14 дней.",'
        '"citations":["returns-01"]}'
    )
    meter = MeteredLLMClient(
        FakeLLMClient(
            [
                LLMResponse(
                    content=(
                        '{"kind":"tool_call",'
                        '"tool_name":"search_policy",'
                        '"arguments":{"query":"Возврат товара"}}'
                    ),
                    model="fake-model",
                    prompt_tokens=25,
                    completion_tokens=10,
                ),
                LLMResponse(
                    content=final_response,
                    model="fake-model",
                    prompt_tokens=25,
                    completion_tokens=10,
                ),
            ]
        ),
        input_price_per_million=1.0,
        output_price_per_million=2.0,
    )
    case = GoldenCase(
        id="gold-executor-005",
        question="Можно ли вернуть товар?",
        type="support",
        input={
            "message": "Можно ли вернуть товар?",
            "session_id": "session-001",
        },
        expected_answer={
            "answer": "Вернуть товар можно в течение 14 дней."
        },
        origin="manual",
    )
    support_registry = RecordingToolRegistry(
    SupportPolicyRegistry()
    )
    executor = MvpCaseExecutor(
        llm=meter,
        attribute_specs=load_attribute_specs(
            Path("data/specs/laptops.yaml")
        ),
        support_registry=support_registry,
    )
    
    prediction = executor.execute(case)

    assert prediction.prediction == {
        "status": "answered",
        "text": "Вернуть товар можно в течение 14 дней.",
        "citations": ["returns-01"],
    }
    assert prediction.raw_response == final_response
    assert prediction.model == "fake-model"
    assert prediction.prompt_tokens == 50
    assert prediction.completion_tokens == 20
    assert prediction.cost_usd == pytest.approx(0.00009)
    assert prediction.tool_calls == ["search_policy"]

def test_executes_content_generation_case() -> None:
    final_response = '{"violations":[]}'
    meter = MeteredLLMClient(
        FakeLLMClient(
            [
                LLMResponse(
                    content=(
                        '{"attributes":{'
                        '"screen_diagonal_in":'
                        '{"value":14.0,"confidence":0.95},'
                        '"processor":'
                        '{"value":"Intel Core i5-1340P",'
                        '"confidence":0.95},'
                        '"ram_gb":{"value":"16","confidence":0.95},'
                        '"storage_gb":{"value":"512","confidence":0.95}'
                        '},'
                        '"unsupported_facts":[]}'
                    ),
                    model="fake-model",
                    prompt_tokens=10,
                    completion_tokens=5,
                ),
                LLMResponse(
                    content=(
                        '{"title":"Ноутбук Lenovo IdeaPad для работы",'
                        '"bullets":["Экран 14 дюймов","ОЗУ 16 ГБ"],'
                        '"description":"Ноутбук Lenovo IdeaPad с ОЗУ 16 ГБ.",'
                        '"keywords":["ноутбук","Lenovo"],'
                        '"used_attributes":{'
                        '"screen_diagonal_in":14.0,'
                        '"processor":"Intel Core i5-1340P",'
                        '"ram_gb":"16",'
                        '"storage_gb":"512"'
                        '}}'
                    ),
                    model="fake-model",
                    prompt_tokens=10,
                    completion_tokens=5,
                ),
                LLMResponse(
                    content='{"unsupported_claims":[]}',
                    model="fake-model",
                    prompt_tokens=10,
                    completion_tokens=5,
                ),
                LLMResponse(
                    content=final_response,
                    model="fake-model",
                    prompt_tokens=10,
                    completion_tokens=5,
                ),
            ]
        ),
        input_price_per_million=1.0,
        output_price_per_million=2.0,
    )
    case = GoldenCase(
        id="gold-executor-006",
        question="Создай карточку.",
        type="content_generation",
        input={
            "product": {
                "sku": "LAP-EVAL-022",
                "category": "laptops",
                "brand": "Lenovo",
                "model": "IdeaPad",
                "price": "75000",
                "sales_count": 0,
                "supplier_description": (
                    "Lenovo IdeaPad, экран 14.0 дюймов, "
                    "процессор Intel Core i5-1340P, "
                    "ОЗУ 16 ГБ, SSD 512 ГБ"
                ),
            }
        },
        expected_answer={},
        origin="manual",
    )
    executor = MvpCaseExecutor(
        llm=meter,
        attribute_specs=load_attribute_specs(
            Path("data/specs/laptops.yaml")
        ),
    )

    prediction = executor.execute(case)

    assert prediction.prediction == {
        "status": "completed",
        "title": "Ноутбук Lenovo IdeaPad для работы",
        "used_attributes": {
            "screen_diagonal_in": 14.0,
            "processor": "Intel Core i5-1340P",
            "ram_gb": "16",
            "storage_gb": "512",
        },
    }
    assert prediction.raw_response == final_response
    assert prediction.model == "fake-model"
    assert prediction.prompt_tokens == 40
    assert prediction.completion_tokens == 20
    assert prediction.cost_usd == pytest.approx(0.00008)