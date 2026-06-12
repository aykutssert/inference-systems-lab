import argparse
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Iterable
from typing import TextIO, cast

DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL = "mlx-community/Qwen3-1.7B-4bit"
DEFAULT_TIMEOUT_SECONDS = 60.0


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
    prompt: str,
    *,
    base_url: str,
    model: str,
    timeout_seconds: float,
    output: TextIO,
) -> None:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

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
                    output.write(content)
                    output.flush()
    except urllib.error.HTTPError as error:
        raise RuntimeError(format_http_error(error)) from error

    output.write("\n")
    output.flush()


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
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()

    print("Enter a prompt. Use Ctrl-D or an empty prompt to exit.")
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except EOFError:
            print()
            return
        if not prompt:
            return

        print("Assistant: ", end="", flush=True)
        try:
            stream_prompt(
                prompt,
                base_url=args.base_url,
                model=args.model,
                timeout_seconds=args.timeout,
                output=sys.stdout,
            )
        except (RuntimeError, urllib.error.URLError) as error:
            print(f"\nError: {error}", file=sys.stderr)


if __name__ == "__main__":  # pragma: no cover
    main()
