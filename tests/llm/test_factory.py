from marketplace_agent.config import Settings
from marketplace_agent.llm.factory import create_llm_client


def test_factory_creates_groq_client() -> None:
    client = create_llm_client(
        Settings(
            groq_model="qwen/qwen3.6-27b",
            llm_provider="groq",
            groq_api_key="groq-key",
            openrouter_api_key="openrouter-key",
        )
    )

    assert client.base_url == "https://api.groq.com/openai/v1"
    assert client.model == "qwen/qwen3.6-27b"
    assert client.reasoning_effort == "none"


def test_factory_creates_openrouter_client() -> None:
    client = create_llm_client(
        Settings(
            llm_provider="openrouter",
            groq_api_key="groq-key",
            openrouter_api_key="openrouter-key",
            openrouter_model="openai/gpt-oss-20b",
        )
    )

    assert client.base_url == "https://openrouter.ai/api/v1"
    assert client.model == "openai/gpt-oss-20b"