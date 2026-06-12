import pytest

from production_serving.config import (
    DEFAULT_FIRST_TOKEN_TIMEOUT_SECONDS,
    first_token_timeout_seconds,
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
