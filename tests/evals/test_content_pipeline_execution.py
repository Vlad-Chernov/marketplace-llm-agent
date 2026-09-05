from marketplace_agent.content.pipeline import run_content_pipeline
from marketplace_agent.evals.content_pipeline import (
    load_content_pipeline_products,
    run_pipeline_version,
)
from marketplace_agent.llm.base import FakeLLMClient, LLMResponse


def response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        model="fake-model",
        prompt_tokens=10,
        completion_tokens=5,
    )


def test_runs_one_pipeline_version_and_collects_usage() -> None:
    product = load_content_pipeline_products(
        "data/gold/content_pipeline_manifest.json"
    )[0]
    client = FakeLLMClient(
        chat_responses=[
            response(
                """
                {
                  "attributes": {
                    "screen_diagonal_in": {
                      "value": 14.0,
                      "confidence": 0.95
                    },
                    "storage_gb": {
                      "value": "512",
                      "confidence": 0.95
                    }
                  },
                  "unsupported_facts": []
                }
                """
            ),
            response(
                """
                {
                  "title": "Ноутбук для работы и учёбы",
                  "bullets": ["Экран 14 дюймов", "SSD 512 ГБ"],
                  "description": "Ноутбук с экраном 14 дюймов и SSD 512 ГБ.",
                  "keywords": ["ноутбук"],
                  "used_attributes": {
                    "screen_diagonal_in": 14.0,
                    "storage_gb": "512"
                  }
                }
                """
            ),
            response('{"unsupported_claims": []}'),
            response('{"violations": []}'),
        ]
    )

    results = run_pipeline_version(
        [product],
        llm_factory=lambda: client,
        pipeline=run_content_pipeline,
        max_attempts=1,
        input_price_per_million=1.0,
        output_price_per_million=2.0,
    )

    assert len(results) == 1
    assert results[0].sku == product.sku
    assert results[0].used_attributes == {
        "screen_diagonal_in": 14.0,
        "storage_gb": "512",
    }
    assert results[0].violations == []
    assert results[0].prompt_tokens == 40
    assert results[0].completion_tokens == 20
    assert results[0].cost_usd == 0.00008
