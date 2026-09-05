import os
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv

LLMProviderName = Literal["groq", "openrouter", "gigachat"]


@dataclass(frozen=True)
class Settings:
    """Store application settings loaded from the environment."""

    llm_provider: LLMProviderName
    groq_api_key: str
    openrouter_api_key: str
    gigachat_authorization_key: str = ""
    groq_model: str = "groq/compound-mini"
    openrouter_model: str = "openrouter/free"
    gigachat_model: str = "GigaChat-2-Pro"
    input_price_per_million: float = 0.0
    output_price_per_million: float = 0.0
    batch_max_workers: int = 2
    batch_llm_requests_per_minute: int = 30

    @property
    def cache_namespace(self) -> str:
        model = {
            "groq": self.groq_model,
            "openrouter": self.openrouter_model,
            "gigachat": self.gigachat_model,
        }[self.llm_provider]
        return f"{self.llm_provider}:{model}"

    @classmethod
    def from_environment(cls) -> "Settings":
        """Load LLM settings from .env and environment variables."""

        load_dotenv()

        provider = os.getenv("LLM_PROVIDER", "groq")
        if provider not in {"groq", "openrouter", "gigachat"}:
            raise ValueError(
                "LLM_PROVIDER must be 'groq', 'openrouter', or 'gigachat'."
            )

        groq_api_key = os.getenv("GROQ_API_KEY", "")
        openrouter_api_key = os.getenv("OPENROUTER_API_KEY", "")
        gigachat_authorization_key = os.getenv(
            "GIGACHAT_AUTHORIZATION_KEY",
            "",
        )
        batch_max_workers = int(os.getenv("BATCH_MAX_WORKERS", "2"))
        batch_llm_requests_per_minute = int(
            os.getenv("BATCH_LLM_REQUESTS_PER_MINUTE", "30")
        )

        if provider == "groq" and not groq_api_key:
            raise ValueError("GROQ_API_KEY is required.")

        if provider == "openrouter" and not openrouter_api_key:
            raise ValueError("OPENROUTER_API_KEY is required.")

        if provider == "gigachat" and not gigachat_authorization_key:
            raise ValueError("GIGACHAT_AUTHORIZATION_KEY is required.")

        if batch_max_workers < 1:
            raise ValueError("BATCH_MAX_WORKERS must be at least 1.")

        if batch_llm_requests_per_minute < 1:
            raise ValueError(
                "BATCH_LLM_REQUESTS_PER_MINUTE must be at least 1."
            )

        return cls(
            llm_provider=provider,
            groq_api_key=groq_api_key,
            openrouter_api_key=openrouter_api_key,
            gigachat_authorization_key=gigachat_authorization_key,
            groq_model=os.getenv("GROQ_MODEL", "groq/compound-mini"),
            openrouter_model=os.getenv(
                "OPENROUTER_MODEL",
                "openrouter/free",
            ),
            gigachat_model=os.getenv(
                "GIGACHAT_MODEL",
                "GigaChat-2-Pro",
            ),
            input_price_per_million=float(
                os.getenv("LLM_INPUT_PRICE_PER_MILLION", "0.0")
            ),
            output_price_per_million=float(
                os.getenv("LLM_OUTPUT_PRICE_PER_MILLION", "0.0")
            ),
            batch_max_workers=batch_max_workers,
            batch_llm_requests_per_minute=(
                batch_llm_requests_per_minute
            ),
        )