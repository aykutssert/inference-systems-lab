import pytest
from pydantic import ValidationError

from service_foundations.config import Settings, get_settings


def test_settings_load_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("APP_ENVIRONMENT", "test")
    monkeypatch.setenv("APP_HOST", "0.0.0.0")
    monkeypatch.setenv("APP_PORT", "9000")
    monkeypatch.setenv("APP_LOG_LEVEL", "warning")

    settings = Settings()

    assert settings.environment == "test"
    assert settings.host == "0.0.0.0"
    assert settings.port == 9000
    assert settings.log_level == "warning"


def test_get_settings_caches_instance() -> None:
    get_settings.cache_clear()

    first = get_settings()
    second = get_settings()

    assert first is second
    get_settings.cache_clear()


@pytest.mark.parametrize("port", ["0", "65536"])
def test_settings_reject_invalid_ports(
    monkeypatch: pytest.MonkeyPatch,
    port: str,
) -> None:
    monkeypatch.setenv("APP_PORT", port)

    with pytest.raises(ValidationError):
        Settings()
