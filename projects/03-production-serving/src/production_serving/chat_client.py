import argparse
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from typing import Literal, TextIO, cast

from local_inference.chat import ChatMessage, build_prompt
from local_inference.config import MODEL_ID, MODEL_REVISION
from transformers import AutoConfig, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL = MODEL_ID
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_TOKENS = 256
WARNING_THRESHOLD = 0.80
COMPACTION_THRESHOLD = 0.90


class ContextBudgetExceededError(Exception):
    pass


def load_tokenizer(model: str, revision: str | None) -> PreTrainedTokenizerBase:
    try:
        return cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(
                model, revision=revision, local_files_only=True
            ),
        )
    except OSError:
        return cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(model, revision=revision),
        )


def resolve_context_window(model: str, revision: str | None) -> int:
    try:
        config = AutoConfig.from_pretrained(
            model, revision=revision, local_files_only=True
        )
    except OSError:
        config = AutoConfig.from_pretrained(model, revision=revision)

    context_window = getattr(config, "max_position_embeddings", None)
    if not isinstance(context_window, int) or context_window <= 0:
        raise RuntimeError(
            f"Model config for '{model}' has no valid max_position_embeddings"
        )
    return context_window


def count_prompt_tokens(
    tokenizer: PreTrainedTokenizerBase,
    messages: Sequence[ChatMessage],
) -> int:
    rendered = build_prompt(tokenizer, messages)
    return len(tokenizer.encode(rendered, add_special_tokens=False))


class ConversationHistory:
    """Per-process conversation history sent in full with every request."""

    def __init__(self, system_prompt: str | None = None) -> None:
        self.system_prompt = system_prompt
        self.turns: list[ChatMessage] = []
        self.memory: list[str] = []

    def add(self, role: Literal["user", "assistant"], content: str) -> None:
        self.turns.append({"role": role, "content": content})

    def remove_pending_user(self) -> None:
        if self.turns and self.turns[-1]["role"] == "user":
            self.turns.pop()

    def messages(self) -> list[ChatMessage]:
        result: list[ChatMessage] = []
        if self.system_prompt is not None:
            result.append({"role": "system", "content": self.system_prompt})
        if self.memory:
            memory = "\n".join(f"- {item}" for item in self.memory)
            result.append(
                {
                    "role": "system",
                    "content": f"Conversation memory:\n{memory}",
                }
            )
        result.extend(self.turns)
        return result

    def input_usage(
        self,
        tokenizer: PreTrainedTokenizerBase,
        context_window: int,
        max_tokens: int,
    ) -> float:
        budget = context_window - max_tokens
        if budget <= 0:
            raise ContextBudgetExceededError(
                "Completion budget leaves no room for the prompt"
            )
        return count_prompt_tokens(tokenizer, self.messages()) / budget

    def compact(
        self,
        tokenizer: PreTrainedTokenizerBase,
        context_window: int,
        max_tokens: int,
    ) -> int:
        """Drop the oldest turns until the prompt fits the context window.

        The system prompt and the pending user message are always preserved.
        Complete user-assistant pairs are removed together. Returns the number
        of messages removed.
        """
        budget = context_window - max_tokens
        if budget <= 0:
            raise ContextBudgetExceededError(
                "Completion budget leaves no room for the prompt"
            )
        if self.input_usage(tokenizer, context_window, max_tokens) < (
            COMPACTION_THRESHOLD
        ):
            return 0

        target = int(budget * WARNING_THRESHOLD)
        removed = 0
        while count_prompt_tokens(tokenizer, self.messages()) > target:
            if len(self.turns) < 3:
                if count_prompt_tokens(tokenizer, self.messages()) > budget:
                    raise ContextBudgetExceededError(
                        "The system prompt and latest user message exceed the input "
                        "budget"
                    )
                break
            if self.turns[0]["role"] != "user" or self.turns[1]["role"] != "assistant":
                raise RuntimeError(
                    "Conversation history contains an invalid turn order"
                )
            self._remember(self.turns[0]["content"])
            del self.turns[:2]
            removed += 2
        return removed

    def _remember(self, content: str) -> None:
        if "remember" not in content.casefold():
            return
        normalized = " ".join(content.split())
        if normalized not in self.memory:
            self.memory.append(normalized)


def parse_sse(lines: Iterable[bytes]) -> Iterable[dict[str, object]]:
    for raw_line in lines:
        line = raw_line.decode("utf-8").strip()
        if not line.startswith("data: "):
            continue
        data = line.removeprefix("data: ")
        if data == "[DONE]":
            return
        yield cast(dict[str, object], json.loads(data))


def stream_prompt(
    messages: Sequence[ChatMessage],
    *,
    base_url: str,
    model: str,
    max_tokens: int,
    timeout_seconds: float,
    output: TextIO,
) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": list(messages),
            "max_completion_tokens": max_tokens,
            "stream": True,
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    response_parts: list[str] = []
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            for event in parse_sse(response):
                choices = event.get("choices")
                if not isinstance(choices, list) or not choices:
                    continue
                choice = choices[0]
                if not isinstance(choice, dict):
                    continue
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    continue
                content = delta.get("content")
                if isinstance(content, str) and content:
                    response_parts.append(content)
                    output.write(content)
                    output.flush()
    except urllib.error.HTTPError as error:
        raise RuntimeError(format_http_error(error)) from error

    output.write("\n")
    output.flush()
    return "".join(response_parts)


def format_http_error(error: urllib.error.HTTPError) -> str:
    try:
        payload = json.loads(error.read().decode("utf-8"))
        detail = payload["error"]
        message = detail["message"]
        code = detail.get("code")
    except (AttributeError, KeyError, TypeError, ValueError):
        return f"HTTP {error.code}: {error.reason}"

    suffix = f" ({code})" if code else ""
    return f"HTTP {error.code}: {message}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream chat completions in a terminal"
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--context-window",
        type=int,
        default=None,
        help="Lower the client compaction threshold for testing",
    )
    parser.add_argument(
        "--system", default=None, help="Optional system prompt kept across turns"
    )
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model, args.revision)
    model_context_window = resolve_context_window(args.model, args.revision)
    if not 1 <= args.max_tokens <= 4096:
        parser.error("--max-tokens must be between 1 and 4096")
    if args.context_window is not None and not (
        args.max_tokens < args.context_window <= model_context_window
    ):
        parser.error(
            "--context-window must be greater than --max-tokens and no greater "
            "than the model context window"
        )
    context_window = args.context_window or model_context_window
    print(
        f"Client context budget: {context_window} tokens "
        f"(model supports {model_context_window}, "
        f"generation reserves {args.max_tokens})"
    )

    history = ConversationHistory(system_prompt=args.system)

    print("Enter a prompt. Use Ctrl-D or an empty prompt to exit.")
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not prompt:
            return

        history.add("user", prompt)
        try:
            usage = history.input_usage(tokenizer, context_window, args.max_tokens)
            if WARNING_THRESHOLD <= usage < COMPACTION_THRESHOLD:
                print(
                    f"[context] warning: input budget is {usage:.0%} full",
                    file=sys.stderr,
                )
            removed = history.compact(tokenizer, context_window, args.max_tokens)
        except ContextBudgetExceededError as error:
            history.remove_pending_user()
            print(f"[context] {error}", file=sys.stderr)
            continue
        if removed:
            print(
                f"[context] compacted {removed} message(s) to fit the "
                f"{context_window}-token context window",
                file=sys.stderr,
            )

        print("Assistant: ", end="", flush=True)
        try:
            response = stream_prompt(
                history.messages(),
                base_url=args.base_url,
                model=args.model,
                max_tokens=args.max_tokens,
                timeout_seconds=args.timeout,
                output=sys.stdout,
            )
            history.add("assistant", response)
        except (RuntimeError, urllib.error.URLError) as error:
            history.remove_pending_user()
            print(f"\nError: {error}", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    main()
