import argparse
import json
import time
import urllib.request
from collections.abc import Iterable

from production_serving.chat_client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    parse_sse,
)


def has_content_token(events: Iterable[dict[str, object]]) -> bool:
    for event in events:
        choices = event.get("choices")
        if not isinstance(choices, list) or not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if isinstance(delta, dict) and delta.get("content"):
            return True
    return False


def run_disconnect(
    *,
    prompt: str,
    base_url: str,
    model: str,
    timeout_seconds: float,
) -> dict[str, object]:
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
    started_at = time.monotonic()

    response = urllib.request.urlopen(request, timeout=timeout_seconds)
    try:
        first_token_received = has_content_token(parse_sse(response))
    finally:
        response.close()

    return {
        "scenario": "client_disconnect",
        "first_token_received": first_token_received,
        "disconnected_after_seconds": time.monotonic() - started_at,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run live inference failure scenarios")
    parser.add_argument("scenario", choices=["disconnect"])
    parser.add_argument("--prompt", default="Explain client disconnect cleanup.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args()

    report = run_disconnect(
        prompt=args.prompt,
        base_url=args.base_url,
        model=args.model,
        timeout_seconds=args.timeout,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
