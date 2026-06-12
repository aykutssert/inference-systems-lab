# Production Serving

The v0.3 project adds production serving controls on top of the v0.2 local
inference package.

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

## Terminal Chat Client

Start the service, then run the streaming client in one or more terminals:

```bash
uv run production-chat
```

Use `--base-url`, `--model`, or `--timeout` to override the defaults. Running
multiple clients makes queueing, backpressure, `429` responses, and streaming
behavior visible during development.

The terminal client will not replace automated load testing. A separate load
runner will record structured concurrency, throughput, time-to-first-token,
total-latency, and error-rate results. Failure tests will cover queue
saturation, timeout, client disconnect, and backend failure while metrics
provide operational evidence.
