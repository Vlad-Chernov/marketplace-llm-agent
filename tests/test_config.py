from marketplace_agent.config import Settings


def test_loads_llm_settings_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "groq-test-key")
    monkeypatch.setenv("GROQ_MODEL", "qwen/qwen3.6-27b")
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-test-key")

    settings = Settings.from_environment()

    assert settings.llm_provider == "groq"
    assert settings.groq_api_key == "groq-test-key"
    assert settings.groq_model == "qwen/qwen3.6-27b"
    assert settings.openrouter_api_key == "openrouter-test-key"