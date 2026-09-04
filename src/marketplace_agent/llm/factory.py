from marketplace_agent.config import Settings
from marketplace_agent.llm.base import LLMClient
from marketplace_agent.llm.providers import (
    GigaChatLLMClient,
    OpenAICompatibleLLMClient,
)


def create_llm_client(settings: Settings) -> LLMClient:
    """Create the configured LLM client."""

    if settings.llm_provider == "groq":
        return OpenAICompatibleLLMClient(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
            model=settings.groq_model,
            reasoning_effort=(
                "none" if settings.groq_model.startswith("qwen/") else None
            ),
        )

    if settings.llm_provider == "gigachat":
        return GigaChatLLMClient(
            authorization_key=settings.gigachat_authorization_key,
            model=settings.gigachat_model,
        )

    return OpenAICompatibleLLMClient(
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        model=settings.openrouter_model,
    )