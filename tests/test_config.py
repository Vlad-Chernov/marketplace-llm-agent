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


def test_rejects_missing_key_for_selected_provider(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key")

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        Settings.from_environment()