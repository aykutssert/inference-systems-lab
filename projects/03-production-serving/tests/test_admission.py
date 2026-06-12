import asyncio
import threading
from collections.abc import Iterator, Sequence

import httpx
import pytest
from local_inference.benchmark_runner import GenerationResult
from local_inference.chat import ChatMessage

from production_serving.admission import AdmissionController, QueueFullError
from production_serving.api import stream_chat_completion
from production_serving.app import create_app
from production_serving.streaming import GenerationChunk


class BlockingBackend:
    model = "test-model"

    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()

    def load(self) -> float:
        return 0.1

    def generate_chat(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
    ) -> GenerationResult:
        self.started.set()
        self.release.wait()
        return GenerationResult(
            response="done",
            finish_reason="stop",
            time_to_first_token_seconds=0.1,
            prompt_tokens=1,
            prompt_tokens_per_second=1.0,
            generation_tokens=1,
            generation_tokens_per_second=1.0,
            peak_memory_gb=1.0,
        )

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
    ) -> Iterator[GenerationChunk]:
        yield GenerationChunk(text="done")


@pytest.mark.anyio
async def test_admission_queues_until_active_request_releases() -> None:
    controller = AdmissionController(max_concurrent=1, max_queued=1)
    first = await controller.acquire()
    acquired = asyncio.Event()

    async def wait_for_slot() -> None:
        second = await controller.acquire()
        acquired.set()
        await second.release()

    task = asyncio.create_task(wait_for_slot())
    await asyncio.sleep(0)

    assert controller.active == 1
    assert controller.waiting == 1
    assert not acquired.is_set()

    await first.release()
    await task

    assert controller.active == 0
    assert controller.waiting == 0


@pytest.mark.anyio
async def test_admission_rejects_when_queue_is_full() -> None:
    controller = AdmissionController(max_concurrent=1, max_queued=0)
    lease = await controller.acquire()

    with pytest.raises(QueueFullError):
        await controller.acquire()

    await lease.release()


@pytest.mark.anyio
async def test_full_inference_queue_returns_openai_error() -> None:
    backend = BlockingBackend()
    app = create_app(backend, max_concurrent=1, max_queued=0)
    transport = httpx.ASGITransport(app=app)
    payload = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "Hello"}],
    }

    async with httpx.AsyncClient(
        transport=transport,
        base_url="http://testserver",
    ) as client:
        first_request = asyncio.create_task(
            client.post("/v1/chat/completions", json=payload)
        )
        await asyncio.to_thread(backend.started.wait)

        rejected = await client.post("/v1/chat/completions", json=payload)
        backend.release.set()
        completed = await first_request

    assert completed.status_code == 200
    assert rejected.status_code == 429
    assert rejected.json()["error"]["code"] == "server_busy"


@pytest.mark.anyio
async def test_stream_disconnect_releases_admission_slot() -> None:
    backend = BlockingBackend()
    controller = AdmissionController(max_concurrent=1, max_queued=0)
    lease = await controller.acquire()
    stream = stream_chat_completion(
        backend,
        [{"role": "user", "content": "Hello"}],
        8,
        lease=lease,
    )

    await anext(stream)
    await stream.aclose()

    assert controller.active == 0
