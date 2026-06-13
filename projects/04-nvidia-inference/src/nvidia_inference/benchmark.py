import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class RequestResult:
    request_id: int
    status: int
    prompt_tokens: int
    completion_tokens: int
    time_to_first_token_seconds: float | None
    time_per_output_token_seconds: float | None
    total_latency_seconds: float
    error_code: str | None


def parse_sse(response: Iterable[bytes]) -> Iterator[dict[str, object]]:
    for raw_line in response:
        line = raw_line.decode("utf-8").strip()
        if not line.startswith("data: "):
            continue
        data = line.removeprefix("data: ")
        if data == "[DONE]":
            break
        yield json.loads(data)


def run_request(
    request_id: int,
    *,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    timeout_seconds: float,
) -> RequestResult:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_at = time.monotonic()
    token_times: list[float] = []
    prompt_tokens = 0
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
                            token_times.append(time.monotonic())
                usage = event.get("usage")
                if isinstance(usage, dict):
                    prompt_tokens = int(usage.get("prompt_tokens", 0))
                    completion_tokens = int(usage.get("completion_tokens", 0))
        status = 200
        error_code = None
    except urllib.error.HTTPError as error:
        status = error.code
        error_code = read_error_code(error)
    except (TimeoutError, urllib.error.URLError):
        status = 0
        error_code = "connection_error"

    finished_at = time.monotonic()
    observed_tokens = completion_tokens or len(token_times)
    return RequestResult(
        request_id=request_id,
        status=status,
        prompt_tokens=prompt_tokens,
        completion_tokens=observed_tokens,
        time_to_first_token_seconds=(
            token_times[0] - started_at if token_times else None
        ),
        time_per_output_token_seconds=calculate_tpot(
            token_times,
            observed_tokens,
        ),
        total_latency_seconds=finished_at - started_at,
        error_code=error_code,
    )


def calculate_tpot(token_times: list[float], completion_tokens: int) -> float | None:
    if len(token_times) < 2 or completion_tokens < 2:
        return None
    return (token_times[-1] - token_times[0]) / (completion_tokens - 1)


def read_error_code(error: urllib.error.HTTPError) -> str:
    try:
        payload = json.loads(error.read().decode("utf-8"))
        return str(payload["error"]["code"] or f"http_{error.code}")
    except (KeyError, TypeError, ValueError):
        return f"http_{error.code}"


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def distribution(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def build_report(
    results: list[RequestResult],
    *,
    model: str,
    concurrency: int,
    wall_time_seconds: float,
) -> dict[str, object]:
    successful = [result for result in results if result.status == 200]
    total_tokens = sum(result.completion_tokens for result in successful)
    return {
        "metadata": {
            "model": model,
            "concurrency": concurrency,
            "requests": len(results),
        },
        "summary": {
            "successful": len(successful),
            "errors": len(results) - len(successful),
            "error_rate": (
                (len(results) - len(successful)) / len(results) if results else 0.0
            ),
            "wall_time_seconds": wall_time_seconds,
            "requests_per_second": (
                len(results) / wall_time_seconds if wall_time_seconds else 0.0
            ),
            "output_tokens_per_second": (
                total_tokens / wall_time_seconds if wall_time_seconds else 0.0
            ),
            "latency_seconds": distribution(
                [result.total_latency_seconds for result in successful]
            ),
            "time_to_first_token_seconds": distribution(
                [
                    result.time_to_first_token_seconds
                    for result in successful
                    if result.time_to_first_token_seconds is not None
                ]
            ),
            "time_per_output_token_seconds": distribution(
                [
                    result.time_per_output_token_seconds
                    for result in successful
                    if result.time_per_output_token_seconds is not None
                ]
            ),
        },
        "requests": [asdict(result) for result in results],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark a vLLM server")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--requests", type=int, default=8)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--prompt",
        default="Explain continuous batching in inference systems.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.requests <= 0 or args.concurrency <= 0 or args.max_tokens <= 0:
        parser.error("--requests, --concurrency, and --max-tokens must be positive")

    started_at = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        futures = [
            executor.submit(
                run_request,
                request_id,
                base_url=args.base_url,
                model=args.model,
                prompt=args.prompt,
                max_tokens=args.max_tokens,
                timeout_seconds=args.timeout,
            )
            for request_id in range(args.requests)
        ]
        results = [future.result() for future in futures]

    report = build_report(
        results,
        model=args.model,
        concurrency=args.concurrency,
        wall_time_seconds=time.monotonic() - started_at,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":  # pragma: no cover
    main()
