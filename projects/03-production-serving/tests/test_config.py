from collections.abc import Callable

import pytest

from production_serving.config import (
    DEFAULT_FIRST_TOKEN_TIMEOUT_SECONDS,
    DEFAULT_MAX_CONCURRENT_REQUESTS,
    DEFAULT_MAX_QUEUED_REQUESTS,
    first_token_timeout_seconds,
    max_concurrent_requests,
    max_queued_requests,
)


def test_first_token_timeout_defaults_to_thirty_seconds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SERVING_FIRST_TOKEN_TIMEOUT_SECONDS", raising=False)

    assert first_token_timeout_seconds() == DEFAULT_FIRST_TOKEN_TIMEOUT_SECONDS


def test_first_token_timeout_reads_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SERVING_FIRST_TOKEN_TIMEOUT_SECONDS", "12.5")

    assert first_token_timeout_seconds() == 12.5


@pytest.mark.parametrize("value", ["0", "-1", "invalid"])
def test_first_token_timeout_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("SERVING_FIRST_TOKEN_TIMEOUT_SECONDS", value)

    with pytest.raises(ValueError):
        first_token_timeout_seconds()


def test_admission_limits_use_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SERVING_MAX_CONCURRENT_REQUESTS", raising=False)
    monkeypatch.delenv("SERVING_MAX_QUEUED_REQUESTS", raising=False)

    assert max_concurrent_requests() == DEFAULT_MAX_CONCURRENT_REQUESTS
    assert max_queued_requests() == DEFAULT_MAX_QUEUED_REQUESTS


def test_admission_limits_read_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SERVING_MAX_CONCURRENT_REQUESTS", "2")
    monkeypatch.setenv("SERVING_MAX_QUEUED_REQUESTS", "4")

    assert max_concurrent_requests() == 2
    assert max_queued_requests() == 4


@pytest.mark.parametrize(
    ("name", "value", "reader"),
    [
        ("SERVING_MAX_CONCURRENT_REQUESTS", "0", max_concurrent_requests),
        ("SERVING_MAX_CONCURRENT_REQUESTS", "invalid", max_concurrent_requests),
        ("SERVING_MAX_QUEUED_REQUESTS", "-1", max_queued_requests),
        ("SERVING_MAX_QUEUED_REQUESTS", "invalid", max_queued_requests),
    ],
)
def test_admission_limits_reject_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    reader: Callable[[], int],
) -> None:
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError):
        reader()
