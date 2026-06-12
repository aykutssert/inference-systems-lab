import httpx
import pytest
from test_streaming import FakeBackend

from production_serving.app import create_app
from production_serving.rate_limit import TokenBucketRateLimiter


@pytest.mark.anyio
async def test_token_bucket_refills_over_time() -> None:
    now = 0.0
    limiter = TokenBucketRateLimiter(
        requests_per_minute=60,
        burst=2,
        clock=lambda: now,
    )

    assert await limiter.allow("client") is True
    assert await limiter.allow("client") is True
    assert await limiter.allow("client") is False

    now = 1.0

    assert await limiter.allow("client") is True


@pytest.mark.anyio
async def test_token_bucket_bounds_tracked_clients() -> None:
    limiter = TokenBucketRateLimiter(
        requests_per_minute=60,
        burst=1,
        max_clients=1,
    )

    assert await limiter.allow("first") is True
    assert await limiter.allow("second") is True
    assert await limiter.allow("first") is True


@pytest.mark.anyio
async def test_rate_limit_returns_openai_error_before_queueing() -> None:
    app = create_app(
        FakeBackend(),
        requests_per_minute=1,
        rate_limit_burst_size=1,
    )
    transport = httpx.ASGITransport(app=app, client=("192.0.2.10", 123))
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        accepted = await client.post("/v1/chat/completions", json=payload)
        rejected = await client.post("/v1/chat/completions", json=payload)

    assert accepted.status_code == 200
    assert rejected.status_code == 429
    assert rejected.json()["error"] == {
        "message": "Rate limit exceeded",
        "type": "rate_limit_error",
        "param": None,
        "code": "rate_limit_exceeded",
    }
