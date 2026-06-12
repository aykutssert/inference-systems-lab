import os

DEFAULT_FIRST_TOKEN_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_CONCURRENT_REQUESTS = 1
DEFAULT_MAX_QUEUED_REQUESTS = 8


def first_token_timeout_seconds() -> float:
    raw_value = os.getenv("SERVING_FIRST_TOKEN_TIMEOUT_SECONDS")
    if raw_value is None:
        return DEFAULT_FIRST_TOKEN_TIMEOUT_SECONDS

    value = float(raw_value)
    if value <= 0:
        raise ValueError("SERVING_FIRST_TOKEN_TIMEOUT_SECONDS must be positive")
    return value


def max_concurrent_requests() -> int:
    return positive_int_env(
        "SERVING_MAX_CONCURRENT_REQUESTS",
        DEFAULT_MAX_CONCURRENT_REQUESTS,
    )


def max_queued_requests() -> int:
    raw_value = os.getenv("SERVING_MAX_QUEUED_REQUESTS")
    if raw_value is None:
        return DEFAULT_MAX_QUEUED_REQUESTS

    value = int(raw_value)
    if value < 0:
        raise ValueError("SERVING_MAX_QUEUED_REQUESTS must not be negative")
    return value


def positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    value = int(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value
