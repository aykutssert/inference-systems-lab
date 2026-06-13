import argparse
import json
import os
import sys
from collections.abc import Iterable
from typing import TextIO, cast

import httpx

DEFAULT_BASE_URL = "https://inference.kernelgallery.com"
DEFAULT_MODEL = "Qwen/Qwen3-1.7B-GPTQ-Int8"


def parse_sse(lines: Iterable[str]) -> Iterable[dict[str, object]]:
    for line in lines:
        if not line.startswith("data: "):
            continue
        data = line.removeprefix("data: ")
        if data == "[DONE]":
            return
        yield cast(dict[str, object], json.loads(data))


def stream_prompt(
    prompt: str,
    *,
    api_key: str,
    base_url: str,
    model: str,
    max_tokens: int,
    output: TextIO,
) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }
    response_parts: list[str] = []

    with (
        httpx.Client(timeout=120.0) as client,
        client.stream(
            "POST",
            f"{base_url.rstrip('/')}/v1/chat/completions",
            headers=headers,
            json=payload,
        ) as response,
    ):
        response.raise_for_status()
        for event in parse_sse(response.iter_lines()):
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

    output.write("\n")
    output.flush()
    return "".join(response_parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream chat completions from the internal inference gateway"
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    api_key = os.getenv("INFERENCE_API_KEY")
    if not api_key:
        parser.error("INFERENCE_API_KEY is required")

    print("Enter a prompt. Use Ctrl-D or an empty prompt to exit.")
    while True:
        try:
            prompt = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not prompt:
            return
        try:
            output = stream_prompt(
                prompt,
                api_key=api_key,
                base_url=args.base_url,
                model=args.model,
                max_tokens=args.max_tokens,
                output=sys.stdout,
            )
        except httpx.HTTPError as error:
            print(f"Error: {error}")
            continue
        if not output:
            print("Error: empty response")
