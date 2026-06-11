from collections.abc import Sequence
from typing import Any, Literal, TypedDict


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


def build_prompt(tokenizer: Any, messages: Sequence[ChatMessage]) -> str:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str):
        raise TypeError("Chat template did not return text")
    return rendered
