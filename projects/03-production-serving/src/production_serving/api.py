import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Iterator, Sequence
from typing import Protocol, cast

import anyio
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from local_inference.api import OpenAIAPIError
from local_inference.api_models import (
    ChatCompletionResponse,
    CompletionChoice,
    CompletionMessage,
    CompletionUsage,
    ModelList,
    ModelObject,
)
from local_inference.benchmark_runner import GenerationResult
from local_inference.chat import ChatMessage
from starlette.types import Receive, Scope, Send

from production_serving.admission import (
    AdmissionController,
    AdmissionLease,
    QueueFullError,
)
from production_serving.context import ContextWindowExceededError
from production_serving.metrics import (
    record_generated_tokens,
    record_time_to_first_token,
)
from production_serving.models import ChatCompletionRequest
from production_serving.rate_limit import TokenBucketRateLimiter
from production_serving.streaming import GenerationChunk

logger = logging.getLogger(__name__)
STREAM_END = object()


class CancellableStreamingResponse(StreamingResponse):
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            close = getattr(self.body_iterator, "aclose", None)
            if close is not None:
                await close()


class StreamingBackend(Protocol):
    model: str

    def load(self) -> float: ...

    def generate_chat(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
    ) -> GenerationResult: ...

    def stream_chat(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
    ) -> Iterator[GenerationChunk]: ...


router = APIRouter(prefix="/v1", tags=["OpenAI compatibility"])


def get_backend(request: Request) -> StreamingBackend:
    return cast(StreamingBackend, request.app.state.backend)


def get_first_token_timeout(request: Request) -> float:
    return cast(float, request.app.state.first_token_timeout_seconds)


def get_admission_controller(request: Request) -> AdmissionController:
    return cast(AdmissionController, request.app.state.admission_controller)


def get_rate_limiter(request: Request) -> TokenBucketRateLimiter:
    return cast(TokenBucketRateLimiter, request.app.state.rate_limiter)


@router.get("/models", response_model=ModelList)
def list_models(request: Request) -> ModelList:
    backend = get_backend(request)
    return ModelList(
        data=[
            ModelObject(
                id=backend.model,
                created=int(request.app.state.started_at),
                owned_by="local",
            )
        ]
    )


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(
    payload: ChatCompletionRequest,
    request: Request,
) -> ChatCompletionResponse | StreamingResponse:
    backend = get_backend(request)
    timeout_seconds = get_first_token_timeout(request)
    if payload.model != backend.model:
        raise OpenAIAPIError(
            f"Model '{payload.model}' is not available",
            error_type="invalid_request_error",
            param="model",
            code="model_not_found",
            status_code=404,
        )

    messages: tuple[ChatMessage, ...] = tuple(
        {
            "role": "system" if message.role == "developer" else message.role,
            "content": message.content,
        }
        for message in payload.messages
    )
    validate_context = getattr(backend, "validate_context", None)
    if validate_context is not None:
        try:
            await anyio.to_thread.run_sync(
                validate_context,
                messages,
                payload.token_limit,
            )
        except ContextWindowExceededError as error:
            raise OpenAIAPIError(
                (
                    f"Prompt uses {error.prompt_tokens} tokens and reserves "
                    f"{error.max_tokens} completion tokens, exceeding the "
                    f"{error.context_window}-token context window"
                ),
                error_type="invalid_request_error",
                param="messages",
                code="context_length_exceeded",
                status_code=400,
            ) from error

    client_id = request.client.host if request.client is not None else "unknown"
    if not await get_rate_limiter(request).allow(client_id):
        raise OpenAIAPIError(
            "Rate limit exceeded",
            error_type="rate_limit_error",
            code="rate_limit_exceeded",
            status_code=429,
        )

    try:
        lease = await get_admission_controller(request).acquire()
    except QueueFullError as error:
        raise OpenAIAPIError(
            "The inference queue is full",
            error_type="server_error",
            code="server_busy",
            status_code=429,
        ) from error

    if payload.stream:
        try:
            stream = backend.stream_chat(messages, payload.token_limit)
            started_at = time.monotonic()
            first_chunk = await anyio.to_thread.run_sync(next_or_end, stream)
            if time.monotonic() - started_at > timeout_seconds:
                await close_stream(stream)
                raise request_timeout_error()
            if first_chunk is STREAM_END:
                await close_stream(stream)
                raise OpenAIAPIError(
                    "The inference backend produced no response",
                    error_type="server_error",
                    code="empty_generation",
                    status_code=503,
                )
            record_time_to_first_token(time.monotonic() - started_at)
        except RuntimeError as error:
            logger.exception("backend_streaming_failed")
            await lease.release()
            raise backend_unavailable_error() from error
        except BaseException:
            await lease.release()
            raise
        return CancellableStreamingResponse(
            stream_chat_completion(
                backend,
                messages,
                payload.token_limit,
                stream=stream,
                first_chunk=cast(GenerationChunk, first_chunk),
                lease=lease,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        try:
            started_at = time.monotonic()
            result = await anyio.to_thread.run_sync(
                backend.generate_chat,
                messages,
                payload.token_limit,
            )
        except RuntimeError as error:
            logger.exception("backend_generation_failed")
            raise backend_unavailable_error() from error
        if time.monotonic() - started_at > timeout_seconds:
            raise request_timeout_error()
        record_time_to_first_token(result.time_to_first_token_seconds)
        record_generated_tokens(result.generation_tokens)
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=backend.model,
            choices=[
                CompletionChoice(
                    index=0,
                    message=CompletionMessage(content=result.response),
                    finish_reason=result.finish_reason,
                )
            ],
            usage=CompletionUsage(
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.generation_tokens,
                total_tokens=result.prompt_tokens + result.generation_tokens,
            ),
        )
    finally:
        await lease.release()


def encode_sse(payload: dict[str, object] | str) -> str:
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def next_or_end[T](iterator: Iterator[T]) -> T | object:
    try:
        return next(iterator)
    except StopIteration:
        return STREAM_END


async def close_stream(stream: Iterator[GenerationChunk]) -> None:
    close = getattr(stream, "close", None)
    if close is not None:
        with anyio.CancelScope(shield=True):
            await anyio.to_thread.run_sync(close)


def request_timeout_error() -> OpenAIAPIError:
    return OpenAIAPIError(
        "The inference backend did not produce a result before the timeout",
        error_type="server_error",
        code="request_timeout",
        status_code=504,
    )


def backend_unavailable_error() -> OpenAIAPIError:
    return OpenAIAPIError(
        "The inference backend is temporarily unavailable",
        error_type="server_error",
        code="backend_unavailable",
        status_code=503,
    )


async def stream_chat_completion(
    backend: StreamingBackend,
    messages: Sequence[ChatMessage],
    max_tokens: int,
    *,
    stream: Iterator[GenerationChunk] | None = None,
    first_chunk: GenerationChunk | None = None,
    lease: AdmissionLease | None = None,
) -> AsyncIterator[str]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    common = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": backend.model,
    }
    resolved_stream = stream or backend.stream_chat(messages, max_tokens)
    try:
        yield encode_sse(
            {
                **common,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": None,
                    }
                ],
            }
        )
        if first_chunk is not None:
            if first_chunk.is_final:
                record_generated_tokens(cast(int, first_chunk.generation_tokens))
            for event in encode_chunk(common, first_chunk):
                yield event
        while True:
            item = await anyio.to_thread.run_sync(next_or_end, resolved_stream)
            if item is STREAM_END:
                break
            chunk = cast(GenerationChunk, item)
            if chunk.is_final:
                record_generated_tokens(cast(int, chunk.generation_tokens))
            for event in encode_chunk(common, chunk):
                yield event
        yield encode_sse("[DONE]")
    finally:
        await close_stream(resolved_stream)
        if lease is not None:
            await lease.release()


def encode_chunk(common: dict[str, object], chunk: GenerationChunk) -> tuple[str, ...]:
    if chunk.text:
        return (
            encode_sse(
                {
                    **common,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": chunk.text},
                            "finish_reason": None,
                        }
                    ],
                }
            ),
        )
    if chunk.is_final:
        return (
            encode_sse(
                {
                    **common,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": chunk.finish_reason,
                        }
                    ],
                }
            ),
            encode_sse(
                {
                    **common,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": chunk.prompt_tokens,
                        "completion_tokens": chunk.generation_tokens,
                        "total_tokens": (
                            cast(int, chunk.prompt_tokens)
                            + cast(int, chunk.generation_tokens)
                        ),
                    },
                }
            ),
        )
    return ()
