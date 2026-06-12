import os

DEFAULT_FIRST_TOKEN_TIMEOUT_SECONDS = 30.0
DEFAULT_MAX_CONCURRENT_REQUESTS = 1
DEFAULT_MAX_QUEUED_REQUESTS = 8
DEFAULT_RATE_LIMIT_REQUESTS_PER_MINUTE = 60
DEFAULT_RATE_LIMIT_BURST = 20
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


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


def rate_limit_requests_per_minute() -> int:
    return positive_int_env(
        "SERVING_RATE_LIMIT_REQUESTS_PER_MINUTE",
        DEFAULT_RATE_LIMIT_REQUESTS_PER_MINUTE,
    )


def rate_limit_burst() -> int:
    return positive_int_env(
        "SERVING_RATE_LIMIT_BURST",
        DEFAULT_RATE_LIMIT_BURST,
    )


def serving_host() -> str:
    return os.getenv("SERVING_HOST", DEFAULT_HOST)


def serving_port() -> int:
    return positive_int_env("SERVING_PORT", DEFAULT_PORT)


def positive_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    value = int(raw_value)
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value
