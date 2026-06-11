import logging
import time
import uuid
from collections.abc import Sequence
from typing import Protocol, cast

from fastapi import APIRouter, Request

from local_inference.api_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionChoice,
    CompletionMessage,
    CompletionUsage,
    ModelList,
    ModelObject,
)
from local_inference.benchmark_runner import GenerationResult
from local_inference.chat import ChatMessage

logger = logging.getLogger(__name__)


class ChatBackend(Protocol):
    model: str

    def load(self) -> float: ...

    def generate_chat(
        self,
        messages: Sequence[ChatMessage],
        max_tokens: int,
    ) -> GenerationResult: ...


router = APIRouter(prefix="/v1", tags=["OpenAI compatibility"])


class OpenAIAPIError(Exception):
    def __init__(
        self,
        message: str,
        *,
        error_type: str,
        param: str | None = None,
        code: str | None = None,
        status_code: int = 400,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.param = param
        self.code = code
        self.status_code = status_code


def get_backend(request: Request) -> ChatBackend:
    return cast(ChatBackend, request.app.state.backend)


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
) -> ChatCompletionResponse:
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
