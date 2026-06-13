from collections.abc import AsyncIterator

import httpx
import pytest
from fastapi.testclient import TestClient

from internal_inference_access.app import create_app

ADMIN_KEY = "admin-key"


class SSEStream(httpx.AsyncByteStream):
    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b'data: {"token":"hello"}\n\n'
        yield b"data: [DONE]\n\n"


def test_admin_key_must_be_distinct_from_user_keys() -> None:
    with pytest.raises(ValueError):
        create_app({"user-a": "same-key"}, admin_api_key="same-key")


@pytest.mark.parametrize(
    ("upstream_status", "expected_status"),
    [(200, 200), (503, 503)],
)
def test_health_endpoints_report_upstream_readiness(
    upstream_status: int,
    expected_status: int,
) -> None:
    def handle_request(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://upstream.test/health"
        assert request.headers["authorization"] == "Bearer upstream-key"
        return httpx.Response(upstream_status, request=request)

    upstream_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handle_request),
    )
    app = create_app(
        {"user-a": "key-a"},
        admin_api_key=ADMIN_KEY,
        upstream_base_url="https://upstream.test",
        upstream_api_key="upstream-key",
        http_client=upstream_client,
    )

    with TestClient(app) as client:
        live = client.get("/health/live")
        ready = client.get("/health/ready")

    assert live.status_code == 200
    assert live.json() == {"status": "ok"}
    assert ready.status_code == expected_status


def test_readiness_fails_when_upstream_is_unreachable() -> None:
    def handle_request(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed", request=request)

    upstream_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handle_request),
    )
    app = create_app(
        {"user-a": "key-a"},
        admin_api_key=ADMIN_KEY,
        upstream_base_url="https://upstream.test",
        http_client=upstream_client,
    )

    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 503
    assert response.json() == {"detail": "Inference upstream is not ready"}


def test_authenticated_request_identifies_user() -> None:
    client = TestClient(create_app({"user-a": "key-a"}, admin_api_key=ADMIN_KEY))

    response = client.get(
        "/v1/whoami",
        headers={"Authorization": "Bearer key-a"},
    )

    assert response.status_code == 200
    assert response.json() == {"user": "user-a"}


def test_missing_key_is_rejected() -> None:
    client = TestClient(create_app({"user-a": "key-a"}, admin_api_key=ADMIN_KEY))

    response = client.get("/v1/whoami")

    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_invalid_key_is_rejected() -> None:
    client = TestClient(create_app({"user-a": "key-a"}, admin_api_key=ADMIN_KEY))

    response = client.get(
        "/v1/whoami",
        headers={"Authorization": "Bearer wrong"},
    )

    assert response.status_code == 401


def test_non_streaming_completion_is_forwarded_without_user_key() -> None:
    def handle_request(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://upstream.test/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer upstream-key"
        assert request.content == b'{"model":"test-model"}'
        return httpx.Response(
            200,
            json={"id": "chatcmpl-test"},
            request=request,
        )

    upstream_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handle_request),
    )
    app = create_app(
        {"user-a": "user-key"},
        admin_api_key=ADMIN_KEY,
        upstream_base_url="https://upstream.test",
        upstream_api_key="upstream-key",
        http_client=upstream_client,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer user-key"},
            json={"model": "test-model"},
        )

    assert response.status_code == 200
    assert response.json() == {"id": "chatcmpl-test"}


def test_streaming_completion_preserves_sse_payload() -> None:
    def handle_request(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/event-stream"},
            stream=SSEStream(),
            request=request,
        )

    upstream_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handle_request),
    )
    app = create_app(
        {"user-a": "user-key"},
        admin_api_key=ADMIN_KEY,
        upstream_base_url="https://upstream.test",
        http_client=upstream_client,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer user-key"},
            json={"model": "test-model", "stream": True},
        )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.content == b'data: {"token":"hello"}\n\ndata: [DONE]\n\n'


def test_invalid_key_does_not_reach_upstream() -> None:
    calls = 0

    def handle_request(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, request=request)

    upstream_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handle_request),
    )
    app = create_app(
        {"user-a": "user-key"},
        admin_api_key=ADMIN_KEY,
        upstream_base_url="https://upstream.test",
        http_client=upstream_client,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer wrong"},
            json={"model": "test-model"},
        )

    assert response.status_code == 401
    assert calls == 0


def test_upstream_connection_failure_returns_503() -> None:
    def handle_request(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed", request=request)

    upstream_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handle_request),
    )
    app = create_app(
        {"user-a": "user-key"},
        admin_api_key=ADMIN_KEY,
        upstream_base_url="https://upstream.test",
        http_client=upstream_client,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer user-key"},
            json={"model": "test-model"},
        )

    assert response.status_code == 503
    assert response.json() == {"detail": "Inference upstream is unavailable"}


def test_rate_limit_is_per_user_and_recorded_in_metrics() -> None:
    calls = 0

    def handle_request(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"id": "ok"}, request=request)

    upstream_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handle_request),
    )
    app = create_app(
        {"user-a": "key-a", "user-b": "key-b"},
        admin_api_key=ADMIN_KEY,
        upstream_base_url="https://upstream.test",
        requests_per_minute=1,
        rate_limit_burst=1,
        http_client=upstream_client,
    )

    with TestClient(app) as client:
        first = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer key-a"},
            json={"model": "test-model"},
        )
        rejected = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer key-a"},
            json={"model": "test-model"},
        )
        other_user = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer key-b"},
            json={"model": "test-model"},
        )
        metrics = client.get("/metrics")

    assert first.status_code == 200
    assert rejected.status_code == 429
    assert other_user.status_code == 200
    assert calls == 2
    assert 'gateway_chat_requests_total{status="200",user="user-a"} 1.0' in metrics.text
    assert (
        'gateway_chat_requests_total{status="rate_limited",user="user-a"} 1.0'
        in metrics.text
    )
    assert 'gateway_chat_requests_total{status="200",user="user-b"} 1.0' in metrics.text


def test_admin_can_revoke_and_restore_one_user() -> None:
    app = create_app(
        {"user-a": "key-a", "user-b": "key-b"},
        admin_api_key=ADMIN_KEY,
    )

    with TestClient(app) as client:
        revoke = client.delete(
            "/admin/users/user-a/access",
            headers={"X-Admin-Key": ADMIN_KEY},
        )
        revoked_user = client.get(
            "/v1/whoami",
            headers={"Authorization": "Bearer key-a"},
        )
        other_user = client.get(
            "/v1/whoami",
            headers={"Authorization": "Bearer key-b"},
        )
        restore = client.put(
            "/admin/users/user-a/access",
            headers={"X-Admin-Key": ADMIN_KEY},
        )
        restored_user = client.get(
            "/v1/whoami",
            headers={"Authorization": "Bearer key-a"},
        )

    assert revoke.json() == {"user": "user-a", "status": "revoked"}
    assert revoked_user.status_code == 401
    assert other_user.status_code == 200
    assert restore.json() == {"user": "user-a", "status": "active"}
    assert restored_user.status_code == 200


def test_admin_endpoints_reject_invalid_key_and_unknown_user() -> None:
    app = create_app(
        {"user-a": "key-a"},
        admin_api_key=ADMIN_KEY,
    )

    with TestClient(app) as client:
        unauthorized = client.delete("/admin/users/user-a/access")
        unknown = client.delete(
            "/admin/users/missing/access",
            headers={"X-Admin-Key": ADMIN_KEY},
        )

    assert unauthorized.status_code == 401
    assert unknown.status_code == 404
