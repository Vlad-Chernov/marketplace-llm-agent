from marketplace_agent.config import Settings
from marketplace_agent.llm.providers import OpenAICompatibleLLMClient


def create_llm_client(settings: Settings) -> OpenAICompatibleLLMClient:
    """Create the configured LLM client."""

    if settings.llm_provider == "groq":
        return OpenAICompatibleLLMClient(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
            model="groq/compound-mini",
        )

    return OpenAICompatibleLLMClient(
        api_key=settings.openrouter_api_key,
        base_url="https://openrouter.ai/api/v1",
        model="openrouter/free",
    )