from marketplace_agent.config import Settings
from marketplace_agent.llm.base import Message
from marketplace_agent.llm.factory import create_llm_client


def main() -> None:
    """Send one small request to the selected LLM provider."""

    settings = Settings.from_environment()
    client = create_llm_client(settings)
    response = client.chat(
        messages=[
            Message(
                role="user",
                content="Ответь одним словом: ОК.",
            )
        ],
        tools=None,
        response_schema=None,
        temperature=0.1,
        max_tokens=128,
    )

    print(f"Provider: {settings.llm_provider}")
    print(f"Model: {response.model}")
    print(f"Response: {response.content}")


if __name__ == "__main__":
    main()