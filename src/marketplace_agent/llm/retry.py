from collections.abc import Callable

BASE_DELAY_SECONDS = 1.0
MAX_DELAY_SECONDS = 8.0


def retry_delay_seconds(
    attempt: int,
    random_uniform: Callable[[float, float], float],
) -> float:
    """Return an exponential backoff delay with full jitter."""

    if attempt < 0:
        raise ValueError("attempt must not be negative.")

    upper_bound = min(
        BASE_DELAY_SECONDS * (2**attempt),
        MAX_DELAY_SECONDS,
    )
    return random_uniform(0.0, upper_bound)