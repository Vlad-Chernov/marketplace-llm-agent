from marketplace_agent.llm.base import FakeLLMClient, LLMResponse, Message


def test_fake_client_returns_configured_chat_response() -> None:
    expected_response = LLMResponse(
        content='{"ram_gb": "16"}',
        model="fake-model",
        prompt_tokens=10,
        completion_tokens=5,
    )
    client = FakeLLMClient(chat_responses=[expected_response])

    response = client.chat(
        messages=[Message(role="user", content="Извлеки RAM из описания.")],
        tools=None,
        response_schema=None,
        temperature=0.0,
        max_tokens=100,
    )

    assert response == expected_response


def test_fake_client_returns_configured_embeddings() -> None:
    client = FakeLLMClient(embeddings=[[0.1, 0.2], [0.3, 0.4]])

    vectors = client.embed(["первый текст", "второй текст"])

    assert vectors == [[0.1, 0.2], [0.3, 0.4]]