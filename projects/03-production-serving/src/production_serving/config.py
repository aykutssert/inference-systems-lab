import os

DEFAULT_FIRST_TOKEN_TIMEOUT_SECONDS = 30.0


def first_token_timeout_seconds() -> float:
    raw_value = os.getenv("SERVING_FIRST_TOKEN_TIMEOUT_SECONDS")
    if raw_value is None:
        return DEFAULT_FIRST_TOKEN_TIMEOUT_SECONDS

    value = float(raw_value)
    if value <= 0:
        raise ValueError("SERVING_FIRST_TOKEN_TIMEOUT_SECONDS must be positive")
    return value
