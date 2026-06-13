import pytest

from internal_inference_access.auth import APIKeyAuthenticator, parse_user_keys


def test_parse_user_keys() -> None:
    assert parse_user_keys('{"user-a":"key-a","user-b":"key-b"}') == {
        "user-a": "key-a",
        "user-b": "key-b",
    }


@pytest.mark.parametrize(
    "raw_value",
    [
        "not-json",
        "[]",
        "{}",
        '{"":"key"}',
        '{"user":""}',
        '{"user-a":"same","user-b":"same"}',
    ],
)
def test_parse_user_keys_rejects_invalid_configuration(raw_value: str) -> None:
    with pytest.raises(ValueError):
        parse_user_keys(raw_value)


def test_authenticator_returns_user_for_matching_key() -> None:
    authenticator = APIKeyAuthenticator({"user-a": "key-a"})

    assert authenticator.authenticate("key-a") == "user-a"
    assert authenticator.authenticate("wrong") is None


def test_authenticator_revokes_and_restores_user() -> None:
    authenticator = APIKeyAuthenticator({"user-a": "key-a"})

    assert authenticator.revoke("user-a") is True
    assert authenticator.authenticate("key-a") is None
    assert authenticator.restore("user-a") is True
    assert authenticator.authenticate("key-a") == "user-a"
    assert authenticator.revoke("missing") is False
    assert authenticator.restore("missing") is False
