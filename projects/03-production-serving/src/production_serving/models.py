from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["developer", "system", "user", "assistant"]
    content: str = Field(min_length=1)


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    max_completion_tokens: int | None = Field(default=None, ge=1, le=4096)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    n: Literal[1] = 1
    stream: bool = False

    @model_validator(mode="after")
    def reject_conflicting_token_limits(self) -> "ChatCompletionRequest":
        if self.max_completion_tokens is not None and self.max_tokens is not None:
            raise ValueError("Use either max_completion_tokens or max_tokens, not both")
        return self

    @property
    def token_limit(self) -> int:
        return self.max_completion_tokens or self.max_tokens or 256
