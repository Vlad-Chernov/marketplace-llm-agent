import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv

LLMProviderName = Literal["groq", "openrouter"]


@dataclass(frozen=True)
class Settings:
    """Store application settings loaded from the environment."""

    llm_provider: LLMProviderName
    groq_api_key: str
    openrouter_api_key: str

    @classmethod
    def from_environment(cls) -> "Settings":
        """Load LLM settings from .env and environment variables."""

        load_dotenv()

        provider = os.getenv("LLM_PROVIDER", "groq")
        if provider not in {"groq", "openrouter"}:
            raise ValueError("LLM_PROVIDER must be 'groq' or 'openrouter'.")

        groq_api_key = os.getenv("GROQ_API_KEY", "")
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")

        if not groq_api_key:
            raise ValueError("GROQ_API_KEY is required.")
        if not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required.")

        return cls(
            llm_provider=provider,
            groq_api_key=groq_api_key,
            openrouter_api_key=openrouter_api_key,
        )