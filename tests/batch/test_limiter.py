from marketplace_agent.batch.limiter import (
    RateLimitedLLMClient,
    RateLimiter,
)
from marketplace_agent.llm.base import FakeLLMClient, LLMResponse, Message


def test_limiter_spaces_requests_by_configured_interval() -> None:
    now = 0.0
    sleeps: list[float] = []

    def clock() -> float:
        return now

    def sleeper(delay: float) -> None:
        nonlocal now
        sleeps.append(delay)
        now += delay

    limiter = RateLimiter(30, clock=clock, sleeper=sleeper)

    limiter.acquire()
    limiter.acquire()
    limiter.acquire()

    assert sleeps == [2.0, 2.0]


def test_rate_limited_client_limits_each_chat_request() -> None:
    now = 0.0
    sleeps: list[float] = []

    def clock() -> float:
        return now

    def sleeper(delay: float) -> None:
        nonlocal now
        sleeps.append(delay)
        now += delay

    response = LLMResponse(
        content="{}",
        model="fake-model",
        prompt_tokens=0,
        completion_tokens=0,
    )
    client = RateLimitedLLMClient(
        FakeLLMClient(chat_responses=[response, response]),
        RateLimiter(60, clock=clock, sleeper=sleeper),
    )
    messages = [Message(role="user", content="Проверка")]

    client.chat(
        messages=messages,
        tools=None,
        response_schema=None,
        temperature=0.0,
        max_tokens=10,
    )
    client.chat(
        messages=messages,
        tools=None,
        response_schema=None,
        temperature=0.0,
        max_tokens=10,
    )

    assert sleeps == [1.0]