from pydantic import BaseModel

from marketplace_agent.config import Settings
from marketplace_agent.llm.base import Message
from marketplace_agent.llm.factory import create_llm_client
from marketplace_agent.llm.structured import StructuredOutputError, chat_structured
from marketplace_agent.llm.providers import LLMProviderError


class LaptopMemory(BaseModel):
    ram_gb: int


def main() -> None:
    """Measure structured JSON validity on five small prompts."""

    client = create_llm_client(Settings.from_environment())
    prompts = [
        "В ноутбуке 8 GB RAM. Верни JSON с полем ram_gb.",
        "В ноутбуке 16 GB RAM. Верни JSON с полем ram_gb.",
        "В ноутбуке 32 GB RAM. Верни JSON с полем ram_gb.",
        "В ноутбуке 64 GB RAM. Верни JSON с полем ram_gb.",
        "В ноутбуке 16 GB RAM. Верни JSON с полем ram_gb.",
    ]
    valid_count = 0

    for index, prompt in enumerate(prompts, start=1):
        try:
            result = chat_structured(
                client=client,
                messages=[Message(role="user", content=prompt)],
                response_schema=LaptopMemory,
                max_retries=1,
            )
            valid_count += 1
            print(f"Case {index}: valid, ram_gb={result.ram_gb}")
        except StructuredOutputError:
            print(f"Case {index}: invalid")
        except LLMProviderError as error:
            print(f"Case {index}: provider error ({error})")

    print(f"Valid responses: {valid_count}/{len(prompts)}")


if __name__ == "__main__":
    main()