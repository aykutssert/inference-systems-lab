import json
import time
from collections.abc import Iterator, Sequence

import anyio
import httpx
from fastapi.testclient import TestClient
from local_inference.benchmark_runner import GenerationResult
from local_inference.chat import ChatMessage
from openai import OpenAI
from starlette.requests import ClientDisconnect
from starlette.types import Message, Scope

from production_serving.api import (
    CancellableStreamingResponse,
    stream_chat_completion,
)
from production_serving.app import create_app
from production_serving.streaming import GenerationChunk


class FakeBackend:
    model = "test-model"

    def load(self) -> float:
        return 0.1

    def generate_chat(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
    ) -> GenerationResult:
        return GenerationResult(
            response="non-streaming response",
            finish_reason="stop",
            time_to_first_token_seconds=0.1,
            prompt_tokens=8,
            prompt_tokens_per_second=100.0,
            generation_tokens=4,
            generation_tokens_per_second=80.0,
            peak_memory_gb=1.1,
        )

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
    ) -> Iterator[GenerationChunk]:
        yield GenerationChunk(text="streaming ")
        yield GenerationChunk(text="response")
        yield GenerationChunk(
            text="",
            finish_reason="stop",
            prompt_tokens=8,
            generation_tokens=4,
        )


class CancellableIterator(Iterator[GenerationChunk]):
    def __init__(self) -> None:
        self.closed = False

    def __next__(self) -> GenerationChunk:
        return GenerationChunk(text="token")

    def close(self) -> None:
        self.closed = True


class CancellableBackend(FakeBackend):
    def __init__(self) -> None:
        self.stream = CancellableIterator()

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
    ) -> Iterator[GenerationChunk]:
        return self.stream


class SlowBackend(FakeBackend):
    def generate_chat(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
    ) -> GenerationResult:
        time.sleep(0.05)
        return super().generate_chat(messages, max_tokens)

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
    ) -> Iterator[GenerationChunk]:
        time.sleep(0.05)
        yield GenerationChunk(text="late")


def test_sse_chunk_order_and_usage() -> None:
    with TestClient(create_app(FakeBackend())) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

    events = [
        line.removeprefix("data: ")
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    payloads = [json.loads(event) for event in events[:-1]]

    assert response.headers["content-type"].startswith("text/event-stream")
    assert payloads[0]["choices"][0]["delta"]["role"] == "assistant"
    assert [payload["choices"][0]["delta"]["content"] for payload in payloads[1:3]] == [
        "streaming ",
        "response",
    ]
    assert payloads[3]["choices"][0]["finish_reason"] == "stop"
    assert payloads[4]["usage"]["total_tokens"] == 12
    assert events[-1] == "[DONE]"


def test_non_streaming_contract_remains_available() -> None:
    with TestClient(create_app(FakeBackend())) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == (
        "non-streaming response"
    )


def test_official_openai_sdk_parses_stream() -> None:
    with TestClient(create_app(FakeBackend())) as server:

        def handle_request(request: httpx.Request) -> httpx.Response:
            response = server.request(
                request.method,
                request.url.raw_path.decode(),
                headers=dict(request.headers),
                content=request.content,
            )
            return httpx.Response(
                status_code=response.status_code,
                headers=response.headers,
                content=response.content,
                request=request,
            )

        with OpenAI(
            api_key="local-test-key",
            base_url="http://testserver/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(handle_request)),
        ) as client:
            chunks = list(
                client.chat.completions.create(
                    model="test-model",
                    messages=[{"role": "user", "content": "Hello"}],
                    stream=True,
                )
            )

    content = "".join(
        chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices
    )
    usage_chunk = next(chunk for chunk in chunks if chunk.usage is not None)

    assert content == "streaming response"
    assert usage_chunk.usage is not None
    assert usage_chunk.usage.total_tokens == 12


def test_stream_cleanup_closes_backend_iterator() -> None:
    backend = CancellableBackend()

    async def consume_and_disconnect() -> None:
        stream = stream_chat_completion(
            backend,
            [{"role": "user", "content": "Hello"}],
            32,
        )
        await anext(stream)
        await anext(stream)
        await stream.aclose()

    anyio.run(consume_and_disconnect)

    assert backend.stream.closed is True


def test_client_disconnect_closes_backend_iterator() -> None:
    backend = CancellableBackend()

    async def disconnect_during_stream() -> None:
        stream = stream_chat_completion(
            backend,
            [{"role": "user", "content": "Hello"}],
            32,
        )
        streaming_response = CancellableStreamingResponse(stream)
        scope: Scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.4"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "raw_path": b"/v1/chat/completions",
            "query_string": b"",
            "root_path": "",
            "headers": [],
            "client": ("127.0.0.1", 1),
            "server": ("127.0.0.1", 8000),
            "state": {},
        }

        async def receive() -> Message:
            return {"type": "http.request", "body": b"", "more_body": False}

        send_calls = 0

        async def send(message: Message) -> None:
            nonlocal send_calls
            if message["type"] == "http.response.body":
                send_calls += 1
                if send_calls == 2:
                    raise OSError("client disconnected")

        try:
            await streaming_response(scope, receive, send)
        except ClientDisconnect:
            pass
        else:
            raise AssertionError("Expected ClientDisconnect")

    anyio.run(disconnect_during_stream)

    assert backend.stream.closed is True


def test_non_streaming_first_result_timeout() -> None:
    with TestClient(create_app(SlowBackend(), timeout_seconds=0.01)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "request_timeout"


def test_streaming_first_token_timeout() -> None:
    with TestClient(create_app(SlowBackend(), timeout_seconds=0.01)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "request_timeout"
