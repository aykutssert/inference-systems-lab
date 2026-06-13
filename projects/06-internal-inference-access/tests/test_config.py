import pytest
from pydantic import ValidationError

from internal_inference_access.config import Settings, get_settings


def test_settings_load_environment_variables(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GATEWAY_API_KEYS", '{"user-a":"key-a"}')
    monkeypatch.setenv("GATEWAY_ADMIN_API_KEY", "admin-key")
    monkeypatch.setenv("GATEWAY_UPSTREAM_BASE_URL", "https://inference.test")
    monkeypatch.setenv("GATEWAY_UPSTREAM_API_KEY", "upstream-key")
    monkeypatch.setenv("GATEWAY_HOST", "0.0.0.0")
    monkeypatch.setenv("GATEWAY_PORT", "9000")
    monkeypatch.setenv("GATEWAY_HEALTH_TIMEOUT_SECONDS", "1.5")
    monkeypatch.setenv("GATEWAY_RATE_LIMIT_REQUESTS_PER_MINUTE", "120")
    monkeypatch.setenv("GATEWAY_RATE_LIMIT_BURST", "20")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.api_keys == {"user-a": "key-a"}
    assert settings.admin_api_key.get_secret_value() == "admin-key"
    assert str(settings.upstream_base_url) == "https://inference.test/"
    assert settings.upstream_api_key == "upstream-key"
    assert settings.host == "0.0.0.0"
    assert settings.port == 9000
    assert settings.health_timeout_seconds == 1.5
    assert settings.rate_limit_requests_per_minute == 120
    assert settings.rate_limit_burst == 20


def test_empty_upstream_key_is_optional(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_API_KEYS", '{"user-a":"key-a"}')
    monkeypatch.setenv("GATEWAY_ADMIN_API_KEY", "admin-key")
    monkeypatch.setenv("GATEWAY_UPSTREAM_BASE_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("GATEWAY_UPSTREAM_API_KEY", "")

    settings = Settings(_env_file=None)  # type: ignore[call-arg]

    assert settings.upstream_api_key is None


def test_required_settings_are_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GATEWAY_API_KEYS", raising=False)
    monkeypatch.delenv("GATEWAY_ADMIN_API_KEY", raising=False)
    monkeypatch.delenv("GATEWAY_UPSTREAM_BASE_URL", raising=False)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


@pytest.mark.parametrize("port", ["0", "65536"])
def test_settings_reject_invalid_port(
    monkeypatch: pytest.MonkeyPatch,
    port: str,
) -> None:
    monkeypatch.setenv("GATEWAY_API_KEYS", '{"user-a":"key-a"}')
    monkeypatch.setenv("GATEWAY_ADMIN_API_KEY", "admin-key")
    monkeypatch.setenv("GATEWAY_UPSTREAM_BASE_URL", "http://127.0.0.1:8000")
    monkeypatch.setenv("GATEWAY_PORT", port)

    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]


def test_get_settings_caches_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GATEWAY_API_KEYS", '{"user-a":"key-a"}')
    monkeypatch.setenv("GATEWAY_ADMIN_API_KEY", "admin-key")
    monkeypatch.setenv("GATEWAY_UPSTREAM_BASE_URL", "http://127.0.0.1:8000")
    get_settings.cache_clear()

    with monkeypatch.context() as context:
        context.chdir("/tmp")
        assert get_settings() is get_settings()

    get_settings.cache_clear()
