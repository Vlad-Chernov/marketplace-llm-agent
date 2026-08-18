import json
from collections.abc import Callable
from time import perf_counter
from typing import Any

from marketplace_agent.evals.models import GoldenCase
from marketplace_agent.evals.runner import EvaluationPrediction
from marketplace_agent.llm.base import LLMClient, Message

ATTRIBUTE_KEYS = (
    "screen_diagonal_in",
    "screen_resolution",
    "processor",
    "ram_gb",
    "storage_gb",
    "graphics_card",
    "operating_system",
    "weight_kg",
    "battery_capacity_wh",
    "color",
    "keyboard_layout",
    "refresh_rate_hz",
)


def build_messages(case: GoldenCase) -> list[Message]:
    if case.type == "attribute_extraction":
        allowed_keys = ", ".join(ATTRIBUTE_KEYS)
        system_prompt = (
            "Ты извлекаешь характеристики ноутбука из текста. "
            "Верни только JSON-объект, без Markdown и пояснений. "
            f"Разрешённые ключи: {allowed_keys}. "
            "Не создавай другие ключи. Неизвестное значение укажи как null. "
            "ram_gb, storage_gb и refresh_rate_hz возвращай строкой. "
            "screen_diagonal_in и weight_kg возвращай числом. "
            "battery_capacity_wh возвращай целым числом."
        )
    else:
        system_prompt = (
            "Ответь на русском языке. Используй только факты из вопроса. "
            "Если данных недостаточно, прямо сообщи об этом. "
            "Верни только корректный JSON."
        )

    return [
        Message(role="system", content=system_prompt),
        Message(role="user", content=case.question),
    ]


def create_case_executor(
    client: LLMClient,
    input_price_per_million: float,
    output_price_per_million: float,
) -> Callable[[GoldenCase], EvaluationPrediction]:
    def execute_case(case: GoldenCase) -> EvaluationPrediction:
        started_at = perf_counter()
        response = client.chat(
            messages=build_messages(case),
            tools=None,
            response_schema=None,
            temperature=0.0,
            max_tokens=128,
        )
        latency_ms = round((perf_counter() - started_at) * 1000)

        try:
            prediction: Any = json.loads(response.content)
        except json.JSONDecodeError:
            prediction = response.content

        cost_usd = (
            response.prompt_tokens / 1_000_000 * input_price_per_million
            + response.completion_tokens / 1_000_000 * output_price_per_million
        )

        return EvaluationPrediction(
            prediction=prediction,
            raw_response=response.content,
            model=response.model,
            latency_ms=latency_ms,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            cost_usd=cost_usd,
        )

    return execute_case