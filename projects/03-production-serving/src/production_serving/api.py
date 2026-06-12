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

from production_serving.models import ChatCompletionRequest
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
def create_chat_completion(
    payload: ChatCompletionRequest,
    request: Request,
) -> ChatCompletionResponse | StreamingResponse:
    backend = get_backend(request)
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
    if payload.stream:
        return CancellableStreamingResponse(
            stream_chat_completion(backend, messages, payload.token_limit),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        result = backend.generate_chat(messages, payload.token_limit)
    except RuntimeError as error:
        logger.exception("backend_generation_failed")
        raise OpenAIAPIError(
            "The inference backend is temporarily unavailable",
            error_type="server_error",
            code="backend_unavailable",
            status_code=503,
        ) from error
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


def encode_sse(payload: dict[str, object] | str) -> str:
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def next_or_end[T](iterator: Iterator[T]) -> T | object:
    try:
        return next(iterator)
    except StopIteration:
        return STREAM_END


async def stream_chat_completion(
    backend: StreamingBackend,
    messages: Sequence[ChatMessage],
    max_tokens: int,
) -> AsyncIterator[str]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    common = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": backend.model,
    }

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

    stream = backend.stream_chat(messages, max_tokens)
    try:
        while True:
            item = await anyio.to_thread.run_sync(next_or_end, stream)
            if item is STREAM_END:
                break
            chunk = cast(GenerationChunk, item)
            if chunk.text:
                yield encode_sse(
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
                )
            if chunk.is_final:
                yield encode_sse(
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
                )
                yield encode_sse(
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
                )
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            with anyio.CancelScope(shield=True):
                await anyio.to_thread.run_sync(close)

    yield encode_sse("[DONE]")
