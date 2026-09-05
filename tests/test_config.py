import pytest

from marketplace_agent.config import Settings


def test_loads_groq_settings_without_openrouter_key(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
    monkeypatch.setenv("GROQ_MODEL", "qwen/qwen3.6-27b")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")

    settings = Settings.from_environment()

    assert settings.llm_provider == "groq"
    assert settings.groq_api_key == "groq-test-key"
    assert settings.groq_model == "qwen/qwen3.6-27b"


def test_loads_openrouter_settings_without_groq_key(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key")

    settings = Settings.from_environment()

    assert settings.llm_provider == "openrouter"
    assert settings.openrouter_api_key == "openrouter-test-key"

def test_loads_openrouter_model(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "openai/gpt-oss-20b")

    settings = Settings.from_environment()

    assert settings.openrouter_model == "openai/gpt-oss-20b"

def test_rejects_missing_key_for_selected_provider(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key")

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        Settings.from_environment()

def test_loads_llm_prices(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
    monkeypatch.setenv("LLM_INPUT_PRICE_PER_MILLION", "0.59")
    monkeypatch.setenv("LLM_OUTPUT_PRICE_PER_MILLION", "0.79")

    settings = Settings.from_environment()

    assert settings.input_price_per_million == 0.59
    assert settings.output_price_per_million == 0.79

def test_builds_cache_namespace_for_selected_provider() -> None:
    settings = Settings(
        llm_provider="openrouter",
        groq_api_key="",
        openrouter_api_key="openrouter-test-key",
        openrouter_model="openai/gpt-oss-20b",
    )

    assert settings.cache_namespace == "openrouter:openai/gpt-oss-20b"

def test_loads_gigachat_settings_without_other_provider_keys(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gigachat")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "")
    monkeypatch.setenv("GIGACHAT_AUTHORIZATION_KEY", "gigachat-test-key")
    monkeypatch.setenv("GIGACHAT_MODEL", "GigaChat-2-Pro")

    settings = Settings.from_environment()

    assert settings.llm_provider == "gigachat"
    assert settings.gigachat_authorization_key == "gigachat-test-key"
    assert settings.gigachat_model == "GigaChat-2-Pro"
    assert settings.cache_namespace == "gigachat:GigaChat-2-Pro"

def test_loads_batch_settings(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
    monkeypatch.setenv("BATCH_MAX_WORKERS", "2")
    monkeypatch.setenv("BATCH_LLM_REQUESTS_PER_MINUTE", "30")

    settings = Settings.from_environment()

    assert settings.batch_max_workers == 2
    assert settings.batch_llm_requests_per_minute == 30