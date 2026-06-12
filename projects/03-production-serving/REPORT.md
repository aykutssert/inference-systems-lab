# v0.3 Production Serving Report

## Outcome

v0.3 adds bounded, observable production-serving behavior around the local MLX
inference backend from v0.2.

Completed controls:

- OpenAI-compatible streaming and non-streaming chat completions
- Client disconnect cancellation and iterator cleanup
- First-result soft deadline
- Bounded inference concurrency and FIFO queueing
- Queue saturation rejection
- Per-client token bucket rate limiting
- Prometheus metrics and a provisioned Grafana dashboard
- Manual terminal client
- Repeatable streaming load runner
- Automated and live failure checks

## Runtime Configuration

| Control | Default | Environment variable |
| --- | --- | --- |
| Active inference requests | 1 | `SERVING_MAX_CONCURRENT_REQUESTS` |
| Queued inference requests | 8 | `SERVING_MAX_QUEUED_REQUESTS` |
| First-result soft deadline | 30 seconds | `SERVING_FIRST_TOKEN_TIMEOUT_SECONDS` |
| Rate limit | 60 requests/minute | `SERVING_RATE_LIMIT_REQUESTS_PER_MINUTE` |
| Rate-limit burst | 20 requests | `SERVING_RATE_LIMIT_BURST` |
| Bind host | `127.0.0.1` | `SERVING_HOST` |
| Bind port | 8000 | `SERVING_PORT` |

## Load Results

Measurements were collected on June 12, 2026 using
`mlx-community/Qwen3-1.7B-4bit`, one active inference slot, and eight queue
slots.

| Requests | Concurrency | Success | 429 | RPS | Token/s | p95 TTFT | p95 latency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10 | 5 | 10 | 0 | 0.258 | 65.97 | 16.29 s | 20.11 s |
| 10 | 10 | 9 | 1 | 0.307 | 70.70 | 27.72 s | 31.14 s |

At concurrency 5, all requests fit within the active and queued capacity. At
concurrency 10, the tenth request exceeded the total capacity and received
`429 server_busy`.

The rising tail latency is expected queue wait time. The MLX backend still runs
one inference at a time, so higher request concurrency does not mean parallel
model execution.

Request-level evidence:

- `benchmarks/load-10x5.json`
- `benchmarks/load-10x10.json`

## Failure Results

Automated tests verify:

- Backend failure returns `503 backend_unavailable`.
- First-result timeout returns `504 request_timeout`.
- Backend failure and timeout release the admission slot.
- Queue saturation returns `429 server_busy`.
- Rate limiting returns `429 rate_limit_exceeded`.
- Streaming disconnect closes the backend iterator and releases the slot.

Live disconnect verification on June 12, 2026:

- Client disconnected after the first content token in 0.50 seconds.
- The immediate follow-up request returned `200`.
- Follow-up time to first token was 0.23 seconds.

## Observability

The Docker Compose observability stack was verified on June 12, 2026:

- Prometheus target `production-serving` reported `up`.
- Prometheus scraped the service `/metrics` endpoint.
- Grafana 13.0.2 provisioned the `Production Serving` dashboard.
- The dashboard contains six panels.

Visible operational signals:

- Request rate and HTTP error rate
- Active requests
- Request duration p50, p95, and p99
- Time to first token p50, p95, and p99
- Generated token throughput

Prometheus and Grafana bind to localhost. The inference service must use
`SERVING_HOST=0.0.0.0` when Prometheus runs in Docker on the same machine.

## Operational Limits

- The timeout is a soft deadline. MLX executes native synchronous work that
  cannot be interrupted safely inside the current process. The active native
  call completes before the API returns `504`.
- Rate limiting identifies the direct socket client. Forwarded IP headers are
  intentionally ignored until a trusted proxy boundary exists.
- Metrics and rate-limit state are process-local. Multi-process or distributed
  serving requires shared coordination.
- The local backend is intentionally single-concurrency. NVIDIA continuous
  batching and GPU concurrency belong to v0.4.

## Completion

Service behavior under load is measured and explained through structured
results. Operational state is visible through a live Prometheus and Grafana
stack. The v0.3 completion criteria are satisfied.
