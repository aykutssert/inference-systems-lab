import io
import json
import sys
import urllib.error
import urllib.request
from email.message import Message
from typing import cast

import pytest
from local_inference.chat import ChatMessage
from local_inference.config import MODEL_ID, MODEL_REVISION
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from production_serving.chat_client import (
    COMPACTION_THRESHOLD,
    WARNING_THRESHOLD,
    ContextBudgetExceededError,
    ConversationHistory,
    count_prompt_tokens,
    format_http_error,
    load_tokenizer,
    parse_sse,
    resolve_context_window,
    stream_prompt,
)


class FakeTokenizer:
    """Minimal tokenizer stand-in: one token per whitespace-separated word."""

    def apply_chat_template(
        self,
        messages: list[ChatMessage],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        return "\n".join(f"{m['role']}: {m['content']}" for m in messages)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return text.split()


def test_parse_sse_yields_json_events_until_done() -> None:
    lines = [
        b": keep-alive\n",
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
        b"\n",
        b'data: {"choices":[],"usage":{"total_tokens":4}}\n',
        b"data: [DONE]\n",
        b'data: {"ignored":true}\n',
    ]

    events = list(parse_sse(lines))

    assert events == [
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [], "usage": {"total_tokens": 4}},
    ]


def test_format_http_error_reads_openai_error() -> None:
    error = urllib.error.HTTPError(
        "http://testserver/v1/chat/completions",
        429,
        "Too Many Requests",
        Message(),
        io.BytesIO(
            json.dumps(
                {
                    "error": {
                        "message": "The inference queue is full",
                        "code": "server_busy",
                    }
                }
            ).encode()
        ),
    )

    assert format_http_error(error) == (
        "HTTP 429: The inference queue is full (server_busy)"
    )


def test_parse_sse_rejects_invalid_json() -> None:
    with pytest.raises(json.JSONDecodeError):
        list(parse_sse([b"data: invalid\n"]))


def test_format_http_error_falls_back_for_invalid_body() -> None:
    error = urllib.error.HTTPError(
        "http://testserver/v1/chat/completions",
        503,
        "Service Unavailable",
        Message(),
        io.BytesIO(b'{"error": "invalid"}'),
    )

    assert format_http_error(error) == "HTTP 503: Service Unavailable"


def test_single_turn_history_has_no_assistant_reply_yet() -> None:
    history = ConversationHistory(system_prompt="You are helpful.")
    history.add("user", "hello")

    assert history.messages() == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "hello"},
    ]


def test_multi_turn_history_preserves_order_and_system_prompt() -> None:
    history = ConversationHistory(system_prompt="You are helpful.")
    history.add("user", "first question")
    history.add("assistant", "first answer")
    history.add("user", "second question")

    assert history.messages() == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "second question"},
    ]


def test_compaction_preserves_explicit_memory_and_recent_turns() -> None:
    tokenizer = cast(PreTrainedTokenizerBase, FakeTokenizer())
    history = ConversationHistory(system_prompt="sys")
    history.add("user", "Remember that my favorite color is blue")
    history.add("assistant", "five six seven eight")
    history.add("user", "nine ten eleven twelve")

    removed = history.compact(tokenizer, context_window=22, max_tokens=2)

    assert removed == 2
    assert history.turns == [{"role": "user", "content": "nine ten eleven twelve"}]
    assert history.memory == ["Remember that my favorite color is blue"]
    assert history.messages()[1] == {
        "role": "system",
        "content": ("Conversation memory:\n- Remember that my favorite color is blue"),
    }


def test_compaction_does_not_run_below_threshold() -> None:
    tokenizer = cast(PreTrainedTokenizerBase, FakeTokenizer())
    history = ConversationHistory()
    history.add("user", "short question")
    history.add("assistant", "short answer")
    history.add("user", "another question")

    removed = history.compact(tokenizer, context_window=100, max_tokens=10)

    assert removed == 0
    assert len(history.turns) == 3


def test_context_thresholds_leave_headroom() -> None:
    assert WARNING_THRESHOLD < COMPACTION_THRESHOLD < 1


def test_compaction_rejects_when_latest_turn_cannot_fit() -> None:
    tokenizer = cast(PreTrainedTokenizerBase, FakeTokenizer())
    history = ConversationHistory(system_prompt="sys")
    history.add("user", "one two three four five six seven eight nine ten")

    with pytest.raises(ContextBudgetExceededError):
        history.compact(tokenizer, context_window=3, max_tokens=1)

    assert len(history.turns) == 1


def test_remove_pending_user_preserves_completed_turns() -> None:
    history = ConversationHistory()
    history.add("user", "first")
    history.add("assistant", "answer")
    history.add("user", "failed")

    history.remove_pending_user()

    assert history.turns == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
    ]


def test_resolve_context_window_for_pinned_model() -> None:
    context_window = resolve_context_window(MODEL_ID, MODEL_REVISION)

    # Verified against the model config: mlx-community/Qwen3-1.7B-4bit pins
    # max_position_embeddings to 40960, not the commonly assumed 32768.
    assert context_window == 40960


def test_count_prompt_tokens_uses_real_tokenizer() -> None:
    tokenizer = load_tokenizer(MODEL_ID, MODEL_REVISION)
    messages: list[ChatMessage] = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello there"},
    ]

    token_count = count_prompt_tokens(tokenizer, messages)

    assert token_count > 0
    assert isinstance(token_count, int)


def test_stream_prompt_sends_full_conversation_history(monkeypatch) -> None:
    captured_requests: list[urllib.request.Request] = []

    class FakeResponse:
        def __init__(self, lines: list[bytes]) -> None:
            self._lines = lines

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def __iter__(self):
            return iter(self._lines)

    def fake_urlopen(request: urllib.request.Request, timeout: float):
        captured_requests.append(request)
        return FakeResponse(
            [
                b'data: {"choices":[{"delta":{"content":"Hi"}}]}\n',
                b'data: {"choices":[{"delta":{"content":" there"}}]}\n',
                b"data: [DONE]\n",
            ]
        )

    import urllib.request as urllib_request

    monkeypatch.setattr(urllib_request, "urlopen", fake_urlopen)

    messages: list[ChatMessage] = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "first reply"},
        {"role": "user", "content": "second"},
    ]
    output = io.StringIO()

    result = stream_prompt(
        messages,
        base_url="http://127.0.0.1:8000",
        model="test-model",
        max_tokens=64,
        timeout_seconds=5.0,
        output=output,
    )

    assert result == "Hi there"
    assert output.getvalue() == "Hi there\n"

    sent_body = json.loads(captured_requests[0].data.decode("utf-8"))
    assert sent_body["messages"] == messages
    assert sent_body["max_completion_tokens"] == 64
    assert sent_body["stream"] is True


@pytest.mark.parametrize("error", [EOFError(), KeyboardInterrupt()])
def test_main_exits_cleanly_on_terminal_interrupt(
    monkeypatch: pytest.MonkeyPatch,
    error: BaseException,
) -> None:
    from production_serving import chat_client

    monkeypatch.setattr(
        sys,
        "argv",
        ["production-chat", "--context-window", "100", "--max-tokens", "64"],
    )
    monkeypatch.setattr(chat_client, "load_tokenizer", lambda model, revision: object())
    monkeypatch.setattr(
        chat_client,
        "resolve_context_window",
        lambda model, revision: 40_960,
    )

    def interrupted_input(prompt: str) -> str:
        raise error

    monkeypatch.setattr("builtins.input", interrupted_input)

    chat_client.main()
