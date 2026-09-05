from marketplace_agent.llm.retry import retry_delay_seconds


def test_retry_delay_uses_exponential_full_jitter() -> None:
    delay = retry_delay_seconds(
        attempt=1,
        random_uniform=lambda start, stop: stop / 2,
    )

    assert delay == 1.0