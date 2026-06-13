import hmac
import json
from typing import Annotated, cast

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import TypeAdapter, ValidationError

USER_KEYS = TypeAdapter(dict[str, str])
bearer_scheme = HTTPBearer(auto_error=False)


def parse_user_keys(raw_value: str) -> dict[str, str]:
    try:
        user_keys = USER_KEYS.validate_python(json.loads(raw_value))
    except (json.JSONDecodeError, ValidationError) as error:
        raise ValueError(
            "GATEWAY_API_KEYS must be a JSON object of user keys"
        ) from error

    if not user_keys or any(not user or not key for user, key in user_keys.items()):
        raise ValueError("GATEWAY_API_KEYS must contain non-empty users and keys")
    if len(set(user_keys.values())) != len(user_keys):
        raise ValueError("GATEWAY_API_KEYS must contain unique keys")
    return user_keys


class APIKeyAuthenticator:
    def __init__(self, user_keys: dict[str, str]) -> None:
        self._user_keys = user_keys
        self._revoked_users: set[str] = set()

    def authenticate(self, presented_key: str) -> str | None:
        for user, expected_key in self._user_keys.items():
            if user not in self._revoked_users and hmac.compare_digest(
                presented_key, expected_key
            ):
                return user
        return None

    def revoke(self, user: str) -> bool:
        if user not in self._user_keys:
            return False
        self._revoked_users.add(user)
        return True

    def restore(self, user: str) -> bool:
        if user not in self._user_keys:
            return False
        self._revoked_users.discard(user)
        return True


def get_authenticator(request: Request) -> APIKeyAuthenticator:
    return cast(APIKeyAuthenticator, request.app.state.authenticator)


def require_user(
    request: Request,
    credentials: Annotated[
        HTTPAuthorizationCredentials | None,
        Depends(bearer_scheme),
    ],
    authenticator: Annotated[APIKeyAuthenticator, Depends(get_authenticator)],
) -> str:
    user = (
        authenticator.authenticate(credentials.credentials)
        if credentials is not None
        else None
    )
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    request.state.authenticated_user = user
    return user
