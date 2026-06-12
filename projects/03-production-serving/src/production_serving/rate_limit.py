import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Bucket:
    tokens: float
    updated_at: float


class TokenBucketRateLimiter:
    def __init__(
        self,
        requests_per_minute: int,
        burst: int,
        *,
        clock: Callable[[], float] = time.monotonic,
        max_clients: int = 10_000,
    ) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        if burst <= 0:
            raise ValueError("burst must be positive")
        if max_clients <= 0:
            raise ValueError("max_clients must be positive")
        self._refill_per_second = requests_per_minute / 60
        self._burst = burst
        self._clock = clock
        self._max_clients = max_clients
        self._buckets: dict[str, Bucket] = {}
        self._lock = asyncio.Lock()

    async def allow(self, client_id: str) -> bool:
        async with self._lock:
            now = self._clock()
            bucket = self._buckets.get(client_id)
            if bucket is None:
                if len(self._buckets) >= self._max_clients:
                    oldest_client = min(
                        self._buckets,
                        key=lambda key: self._buckets[key].updated_at,
                    )
                    del self._buckets[oldest_client]
                bucket = Bucket(tokens=float(self._burst), updated_at=now)
                self._buckets[client_id] = bucket
            else:
                elapsed = max(0.0, now - bucket.updated_at)
                bucket.tokens = min(
                    float(self._burst),
                    bucket.tokens + elapsed * self._refill_per_second,
                )
                bucket.updated_at = now

            if bucket.tokens < 1:
                return False
            bucket.tokens -= 1
            return True
