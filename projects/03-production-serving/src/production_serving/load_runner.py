import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

from production_serving.chat_client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    parse_sse,
)


@dataclass(frozen=True)
class RequestResult:
    request_id: int
    status: int
    time_to_first_token_seconds: float | None
    total_latency_seconds: float
    completion_tokens: int
    error_code: str | None


def run_request(
    request_id: int,
    *,
    prompt: str,
    base_url: str,
    model: str,
    timeout_seconds: float,
) -> RequestResult:
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
    first_token_at: float | None = None
    completion_tokens = 0

    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            for event in parse_sse(response):
                choices = event.get("choices")
                if isinstance(choices, list) and choices:
                    choice = choices[0]
                    if isinstance(choice, dict):
                        delta = choice.get("delta")
                        if isinstance(delta, dict) and delta.get("content"):
                            first_token_at = first_token_at or time.monotonic()
                usage = event.get("usage")
                if isinstance(usage, dict):
                    completion_tokens = cast(int, usage.get("completion_tokens", 0))
        status = 200
        error_code = None
    except urllib.error.HTTPError as error:
        status = error.code
        error_code = read_error_code(error)
    except urllib.error.URLError:
        status = 0
        error_code = "connection_error"
    except TimeoutError:
        status = 0
        error_code = "request_timeout"

    finished_at = time.monotonic()
    return RequestResult(
        request_id=request_id,
        status=status,
        time_to_first_token_seconds=(
            first_token_at - started_at if first_token_at is not None else None
        ),
        total_latency_seconds=finished_at - started_at,
        completion_tokens=completion_tokens,
        error_code=error_code,
    )


def read_error_code(error: urllib.error.HTTPError) -> str:
    try:
        payload = json.loads(error.read().decode("utf-8"))
        return cast(str, payload["error"]["code"] or f"http_{error.code}")
    except (KeyError, TypeError, ValueError):
        return f"http_{error.code}"


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return (
        ordered[lower_index] + (ordered[upper_index] - ordered[lower_index]) * fraction
    )


def build_report(
    results: list[RequestResult],
    *,
    concurrency: int,
    wall_time_seconds: float,
) -> dict[str, object]:
    successful = [result for result in results if result.status == 200]
    latencies = [result.total_latency_seconds for result in successful]
    ttfts = [
        result.time_to_first_token_seconds
        for result in successful
        if result.time_to_first_token_seconds is not None
    ]
    total_tokens = sum(result.completion_tokens for result in successful)
    status_counts: dict[str, int] = {}
    for result in results:
        key = str(result.status)
        status_counts[key] = status_counts.get(key, 0) + 1
    return {
        "summary": {
            "requests": len(results),
            "concurrency": concurrency,
            "successful": len(successful),
            "errors": len(results) - len(successful),
            "error_rate": (
                (len(results) - len(successful)) / len(results) if results else 0.0
            ),
            "status_counts": status_counts,
            "wall_time_seconds": wall_time_seconds,
            "requests_per_second": (
                len(results) / wall_time_seconds if wall_time_seconds else 0.0
            ),
            "completion_tokens_per_second": (
                total_tokens / wall_time_seconds if wall_time_seconds else 0.0
            ),
            "latency_seconds": distribution(latencies),
            "time_to_first_token_seconds": distribution(ttfts),
        },
        "requests": [asdict(result) for result in results],
    }


def distribution(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run streaming inference load")
    parser.add_argument("--requests", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--prompt", default="Explain inference backpressure briefly.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.requests <= 0 or args.concurrency <= 0:
        parser.error("--requests and --concurrency must be positive")

    started_at = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        futures = [
            executor.submit(
                run_request,
                request_id,
                prompt=args.prompt,
                base_url=args.base_url,
                model=args.model,
                timeout_seconds=args.timeout,
            )
            for request_id in range(args.requests)
        ]
        results = [future.result() for future in futures]
    wall_time_seconds = time.monotonic() - started_at

    report = build_report(
        results,
        concurrency=args.concurrency,
        wall_time_seconds=wall_time_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
