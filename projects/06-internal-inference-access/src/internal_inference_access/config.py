from functools import lru_cache

from pydantic import Field, HttpUrl, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from internal_inference_access.auth import parse_user_keys


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="GATEWAY_",
        extra="ignore",
    )

    api_keys: dict[str, str]
    admin_api_key: SecretStr
    upstream_base_url: HttpUrl
    upstream_api_key: str | None = None
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    health_timeout_seconds: float = Field(default=2.0, gt=0)
    rate_limit_requests_per_minute: int = Field(default=60, ge=1)
    rate_limit_burst: int = Field(default=10, ge=1)

    @field_validator("api_keys", mode="before")
    @classmethod
    def validate_api_keys(cls, value: object) -> object:
        if isinstance(value, str):
            return parse_user_keys(value)
        return value

    @field_validator("upstream_api_key", mode="before")
    @classmethod
    def empty_upstream_key_is_none(cls, value: object) -> object:
        return None if value == "" else value


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
