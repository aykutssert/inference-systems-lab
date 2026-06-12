# Production Serving

The v0.3 project adds production serving controls on top of the v0.2 local
inference package.

Status: complete. See [REPORT.md](REPORT.md) for measured behavior, failure
evidence, observability verification, and operational limits.

v0.3.1 context-safety hardening is complete. The pinned model revision
declares a 40,960-token context window. The server counts the rendered chat
prompt with the loaded tokenizer and returns `400 context_length_exceeded`
before generation when the prompt plus completion budget cannot fit.

v0.3.2 conversation memory is complete. The client keeps raw history until
90 percent input usage, warns at 80 percent, and then compacts complete
conversation pairs. Explicit `remember` statements move into a compact memory
message when their source turn is removed.

## Setup

```bash
uv sync --locked
```

## Run

```bash
uv run production-serving
```

The service listens on `http://127.0.0.1:8000`.

## First Slice

The first slice adds Server-Sent Events streaming to
`POST /v1/chat/completions` while preserving the non-streaming response.

The streaming contract includes:

- Initial assistant role chunk
- Ordered content chunks
- Finish reason chunk
- Final token usage chunk
- `[DONE]` terminator
- Official OpenAI Python SDK compatibility
- Backend iterator cleanup after client disconnect

## Real MLX Verification

The pinned `mlx-community/Qwen3-1.7B-4bit` model was verified through the
running v0.3 service on June 12, 2026:

- First content chunk: 1.296 seconds
- Content chunks: 24
- Completion tokens: 24
- Final order: finish reason, usage, `[DONE]`
- Non-streaming completion remained compatible

Client disconnect closes the active backend iterator. MLX generation cleanup is
shielded from request cancellation and runs outside the event loop.

## First Result Timeout

Set the maximum wait for a non-streaming result or the first streaming token:

```bash
SERVING_FIRST_TOKEN_TIMEOUT_SECONDS=30 uv run production-serving
```

The default is 30 seconds. This is a soft deadline because MLX generation runs
in a native synchronous call. The active call finishes safely before the
iterator closes and the API returns an OpenAI-shaped `504` error with code
`request_timeout`.

## Admission Control

Inference concurrency and queue capacity are bounded:

```bash
SERVING_MAX_CONCURRENT_REQUESTS=1 \
SERVING_MAX_QUEUED_REQUESTS=8 \
uv run production-serving
```

The defaults allow one active inference and eight queued requests. Streaming
requests hold their slot until completion or disconnect. A request arriving
after the queue is full receives an OpenAI-shaped `429` error with code
`server_busy`.

## Metrics

Prometheus metrics are available at `GET /metrics`:

- Request count, status, duration, and active requests
- Chat completion time to first token
- Generated completion tokens

Unmatched URLs share one bounded path label to prevent untrusted paths from
creating unlimited metric series.

The Prometheus and Grafana stack is in `observability/`. Its README documents
the host binding, startup command, provisioned dashboard, and metric queries.
The stack was verified live on June 12, 2026 with the Prometheus target up and
the six-panel Grafana dashboard provisioned.

## Terminal Chat Client

Start the service, then run the streaming client in one or more terminals:

```bash
uv run production-chat
```

Use `--base-url`, `--model`, or `--timeout` to override the defaults. Running
multiple clients makes queueing, backpressure, `429` responses, and streaming
behavior visible during development.

For a fast compaction check without changing the server or model limit, lower
only the client threshold:

```bash
uv run production-chat --context-window 100 --max-tokens 64
```

The real tokenizer is still used. The override cannot exceed the model's
40,960-token context window.

Live verification on June 12, 2026 used a 100-token client threshold with 64
tokens reserved for generation. The client removed complete user-assistant
pairs, reported each compaction, and the model no longer recalled information
from removed turns. The server retained its real 40,960-token limit.

The terminal client will not replace automated load testing. A separate load
runner will record structured concurrency, throughput, time-to-first-token,
total-latency, and error-rate results. Failure tests will cover queue
saturation, timeout, client disconnect, and backend failure while metrics
provide operational evidence.

## Load Runner

Run concurrent streaming requests and save structured results:

```bash
uv run production-load \
  --requests 10 \
  --concurrency 5 \
  --output benchmarks/load-10x5.json
```

The report includes per-request status, error code, time to first token, total
latency, completion tokens, and aggregate p50, p95, p99, request throughput,
and token throughput.

## Measured Load Behavior

The pinned MLX model was tested on June 12, 2026 with one active inference slot
and eight queue slots:

| Requests | Concurrency | Success | 429 | p95 TTFT | p95 latency |
| --- | --- | --- | --- | --- | --- |
| 10 | 5 | 10 | 0 | 16.29 s | 20.11 s |
| 10 | 10 | 9 | 1 | 27.72 s | 31.14 s |

At concurrency 5, all requests fit within the active and queued capacity. At
concurrency 10, one request exceeded the total admission capacity and received
`429 server_busy`. The higher tail latency is queue wait time, not parallel
model execution.

Structured request-level results are stored in `benchmarks/`.

## Rate Limiting

Requests are limited per direct client IP before they enter the inference
queue:

```bash
SERVING_RATE_LIMIT_REQUESTS_PER_MINUTE=60 \
SERVING_RATE_LIMIT_BURST=20 \
uv run production-serving
```

The defaults allow a burst of 20 requests and refill one request per second.
Rejected requests receive an OpenAI-shaped `429` error with code
`rate_limit_exceeded`. Forwarded IP headers are not trusted by default.

## Failure Runner

Close a live streaming response immediately after the first content token:

```bash
uv run production-failure disconnect
```

This exercises the server's client-disconnect cleanup path against the real
backend. The server should close the active generation iterator and release its
admission slot so the next request can run.

Live verification on June 12, 2026 disconnected after the first content token
in 0.50 seconds. A follow-up inference started immediately, returned `200`, and
reported a 0.23-second time to first token, confirming that the admission slot
was released.
