from collections.abc import Iterator, Sequence

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import InternalServerError, NotFoundError, OpenAI

from local_inference.app import create_app
from local_inference.benchmark_runner import GenerationChunk, GenerationResult
from local_inference.chat import ChatMessage


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
            response="SDK compatibility works.",
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
        yield GenerationChunk(text="SDK ")
        yield GenerationChunk(text="streaming works.")
        yield GenerationChunk(
            text="",
            finish_reason="stop",
            prompt_tokens=8,
            generation_tokens=4,
        )


class FailingBackend(FakeBackend):
    def generate_chat(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
    ) -> GenerationResult:
        raise RuntimeError("sensitive backend failure")


def test_official_openai_sdk_uses_local_api() -> None:
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
            models = client.models.list()
            completion = client.chat.completions.create(
                model="test-model",
                messages=[
                    {"role": "developer", "content": "Answer briefly."},
                    {"role": "user", "content": "Confirm compatibility."},
                ],
                max_completion_tokens=32,
            )

    assert models.data[0].id == "test-model"
    assert completion.object == "chat.completion"
    assert completion.choices[0].message.content == "SDK compatibility works."
    assert completion.usage is not None
    assert completion.usage.total_tokens == 12


def test_official_openai_sdk_parses_streaming_chunks() -> None:
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
            stream = client.chat.completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "Confirm streaming."}],
                max_completion_tokens=32,
                stream=True,
            )
            chunks = list(stream)

    content = "".join(
        chunk.choices[0].delta.content or "" for chunk in chunks if chunk.choices
    )
    usage_chunk = next(chunk for chunk in chunks if chunk.usage is not None)

    assert content == "SDK streaming works."
    assert usage_chunk.usage is not None
    assert usage_chunk.usage.total_tokens == 12


def test_official_openai_sdk_parses_local_api_errors() -> None:
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

        with (
            OpenAI(
                api_key="local-test-key",
                base_url="http://testserver/v1",
                http_client=httpx.Client(transport=httpx.MockTransport(handle_request)),
            ) as client,
            pytest.raises(NotFoundError) as error_info,
        ):
            client.chat.completions.create(
                model="missing-model",
                messages=[{"role": "user", "content": "Hello"}],
            )

    assert error_info.value.status_code == 404
    assert error_info.value.code == "model_not_found"
    assert error_info.value.param == "model"


def test_official_openai_sdk_parses_backend_failure() -> None:
    with TestClient(create_app(FailingBackend())) as server:

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

        with (
            OpenAI(
                api_key="local-test-key",
                base_url="http://testserver/v1",
                http_client=httpx.Client(transport=httpx.MockTransport(handle_request)),
                max_retries=0,
            ) as client,
            pytest.raises(InternalServerError) as error_info,
        ):
            client.chat.completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "Hello"}],
            )

    assert error_info.value.status_code == 503
    assert error_info.value.code == "backend_unavailable"
    assert "sensitive backend failure" not in str(error_info.value)
