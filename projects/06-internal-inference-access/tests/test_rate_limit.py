import pytest

from internal_inference_access.rate_limit import UserRateLimiter


@pytest.mark.anyio
async def test_user_buckets_are_independent_and_refill() -> None:
    now = 0.0
    limiter = UserRateLimiter(
        requests_per_minute=60,
        burst=1,
        clock=lambda: now,
    )

    assert await limiter.allow("user-a") is True
    assert await limiter.allow("user-a") is False
    assert await limiter.allow("user-b") is True

    now = 1.0

    assert await limiter.allow("user-a") is True


@pytest.mark.parametrize(
    ("requests_per_minute", "burst"),
    [(0, 1), (1, 0)],
)
def test_invalid_limits_are_rejected(
    requests_per_minute: int,
    burst: int,
) -> None:
    with pytest.raises(ValueError):
        UserRateLimiter(requests_per_minute, burst)
