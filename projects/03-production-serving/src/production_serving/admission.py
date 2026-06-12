import asyncio


class QueueFullError(Exception):
    pass


class AdmissionLease:
    def __init__(self, controller: "AdmissionController") -> None:
        self._controller = controller
        self._released = False

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self._controller.release()


class AdmissionController:
    def __init__(self, max_concurrent: int, max_queued: int) -> None:
        if max_concurrent <= 0:
            raise ValueError("max_concurrent must be positive")
        if max_queued < 0:
            raise ValueError("max_queued must not be negative")
        self._max_concurrent = max_concurrent
        self._max_queued = max_queued
        self._active = 0
        self._waiting = 0
        self._condition = asyncio.Condition()

    @property
    def active(self) -> int:
        return self._active

    @property
    def waiting(self) -> int:
        return self._waiting

    async def acquire(self) -> AdmissionLease:
        async with self._condition:
            if self._active < self._max_concurrent and self._waiting == 0:
                self._active += 1
                return AdmissionLease(self)
            if self._waiting >= self._max_queued:
                raise QueueFullError

            self._waiting += 1
            acquired = False
            try:
                await self._condition.wait_for(
                    lambda: self._active < self._max_concurrent
                )
                self._active += 1
                acquired = True
                return AdmissionLease(self)
            finally:
                self._waiting -= 1
                if not acquired and self._active < self._max_concurrent:
                    self._condition.notify(1)

    async def release(self) -> None:
        async with self._condition:
            if self._active <= 0:
                raise RuntimeError("admission lease released without an active request")
            self._active -= 1
            self._condition.notify(1)
