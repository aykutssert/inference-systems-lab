# Roadmap

The roadmap is versioned around working systems, measurable behavior, and
explicit completion criteria.

## v0.1 - Service Foundations

Build a production-oriented Python service without a model dependency.

Scope:

- Python project and dependency management
- HTTP and asynchronous service fundamentals
- Environment-based configuration
- Structured logging
- Liveness and readiness endpoints
- Graceful startup and shutdown
- Docker and Docker Compose
- Unit and integration tests
- Continuous integration

Completion criteria:

- The service starts with one command.
- Configuration is controlled through environment variables.
- Liveness and readiness endpoints report distinct states.
- Shutdown signals are handled cleanly.
- The container image builds and runs.
- Formatting, linting, type checks, tests, and review are clean.

## v0.2 - Local Inference

Run a small open-weight model on Apple Silicon and expose an OpenAI-compatible
API.

Selected model family:

- Source model: `Qwen/Qwen3-1.7B`
- MLX artifact: `mlx-community/Qwen3-1.7B-4bit`
- MLX revision: `3b1b1768f8f8cf8351c712464f906e86c2b8269e`
- GGUF artifact: `Qwen/Qwen3-1.7B-GGUF`
- License: Apache 2.0
- Initial generation mode: non-thinking

Qwen3-1.7B provides a practical balance between fast development and realistic
inference behavior on a 16 GB Apple Silicon machine. The same model family is
available for both MLX and llama.cpp, which allows runtime comparisons without
changing the underlying workload. Non-thinking mode keeps output behavior and
generation cost more predictable during the first benchmarks.

Implementation order:

- Implement the first local backend with MLX.
- Expose generation through an OpenAI-compatible API.
- Add a GGUF backend with llama.cpp.

Runtime comparison is deferred. The available artifacts use different
quantization levels:

- MLX: 4-bit
- GGUF: Q8_0

A direct benchmark would measure the runtime and quantization combination, not
the runtime alone. Both backends are validated independently. Comparison will
resume only when equivalent quantization and a shared process-level memory
metric are available.

The model is a controlled infrastructure workload in this version, not the
product being evaluated. A larger model would increase memory use, startup
time, benchmark duration, and out-of-memory risk without changing the service
architecture being learned. Larger models become useful in later versions when
memory pressure, quantization, KV cache behavior, and GPU scaling are explicit
experiment goals.

Measure:

- Model load time
- Time to first token
- Output tokens per second
- Memory use
- Behavior across context lengths

Completion criteria:

- The service runs locally with one command.
- Benchmarks are repeatable and stored as structured results.

Status: complete.

## v0.3 - Production Serving

Add the service controls required under concurrent load.

First implementation task:

- Add non-buffered Server-Sent Events streaming to
  `POST /v1/chat/completions`.
- Preserve the current non-streaming response contract.
- Verify chunk ordering, final usage, disconnect cleanup, and OpenAI SDK
  compatibility before adding timeouts or queueing.

Scope:

- Streaming
- Request validation
- Timeouts and cancellation
- Rate limiting
- Queueing and backpressure
- Metrics and dashboards
- Load and failure testing

Validation order:

- Add a small streaming terminal client for manual multi-terminal checks.
- Add a repeatable load runner that records structured results, including
  concurrency, throughput, time to first token, total latency, and error rate.
- Exercise queue saturation, `429` rejection, timeout, client disconnect, and
  backend failure scenarios.
- Correlate load-test results with Prometheus metrics and dashboards.

The terminal client is a manual behavior demo, not a benchmark. Automated load
and failure tests remain the source of repeatable performance evidence.

Completion criteria:

- Service behavior under load is measured and explained.
- Operational state is visible through dashboards.

Evidence:

- Structured load results are stored in
  `projects/03-production-serving/benchmarks/`.
- Failure behavior and operational limits are documented in
  `projects/03-production-serving/REPORT.md`.
- Prometheus scraping and the six-panel Grafana dashboard were verified live
  on June 12, 2026.

Status: complete.

## v0.3.1 - Context Safety

Harden multi-turn chat behavior before moving the workload to NVIDIA.

Scope:

- Per-terminal conversation history
- Token-budget visibility
- Deterministic removal of the oldest conversation turns
- Server-side context-window validation
- OpenAI-compatible `context_length_exceeded` errors

The pinned `mlx-community/Qwen3-1.7B-4bit` revision declares a 40,960-token
context window in `config.json`. Runtime validation uses the loaded model config
and the real chat-template token count instead of a hardcoded limit.

Completion criteria:

- Multi-turn terminal conversations preserve recent history.
- The client compacts old turns before exceeding its input budget.
- The server rejects oversized requests before native generation begins.
- Context safety behavior is covered by tests and documented.

Evidence:

- The pinned model config was verified at 40,960 tokens.
- Client compaction was verified live with a reduced client-only threshold.
- Removed turns were no longer available to the model.
- Server-side oversized requests return `400 context_length_exceeded`.

Status: complete.

## v0.3.2 - Conversation Memory

Preserve explicit user memory while avoiding unnecessary compaction.

Scope:

- Keep raw history while context pressure is low.
- Warn when the input budget reaches 80 percent.
- Compact complete user-assistant pairs at 90 percent.
- Preserve explicit `remember` statements in a compact memory message.
- Keep recent turns verbatim.
- Use deterministic deletion as the final fallback.

Completion criteria:

- No compaction occurs below the configured pressure threshold.
- Explicit remembered information survives removal of its source turn.
- Recent conversation turns remain in their original form.
- Continued prompts work after repeated compaction.

Evidence:

- Raw history remains unchanged below the 90 percent compaction threshold.
- Input usage warnings begin at 80 percent.
- Complete user-assistant pairs are removed together.
- Explicit `remember` statements move into the memory system message.
- Deterministic fallback rejects a latest turn that cannot fit safely.

Status: complete.

## v0.4 - NVIDIA Inference

Move the same workload to a rented NVIDIA GPU environment.

Scope:

- CUDA and VRAM behavior
- vLLM
- KV cache
- Continuous batching
- FP16 and GPTQ
- Out-of-memory diagnosis

Measure:

- Time to first token
- Time per output token
- Token throughput
- p50, p95, and p99 latency
- VRAM use
- Concurrent request capacity

Completion criteria:

- Benchmark results explain why each configuration is faster, slower, or
  unable to fit in memory.

Status: complete. The FP16 model ran with vLLM on a rented RunPod
instance using an NVIDIA RTX PRO 4000 Blackwell 24 GB GPU. The local benchmark
client reaches the remote OpenAI-compatible endpoint through an SSH tunnel.
Health, streaming, model metadata, KV cache allocation, and a concurrency 1-8
baseline are verified. Warm output throughput scaled from 114.96 to 849.13
tokens per second for FP16. Official GPTQ Int8 increased throughput by 32.8%
to 42.9%, reduced model-weight memory from 3.22 GiB to 1.92 GiB, and increased
available KV cache from 17.16 GiB to 18.18 GiB. Capacity testing identified
concurrency 32 as the balanced interactive operating point. A controlled CUDA
out-of-memory failure remained isolated to the allocating process, and the
vLLM service continued serving requests.

## v0.5 - Reliable Deployment

Deploy and operate the inference service as a recoverable system.

Scope:

- Kubernetes
- CI/CD
- Model artifact management
- Readiness and liveness probes
- Rollout and rollback
- Resource limits
- Secrets and access control

Completion criteria:

- Failed instances recover automatically.
- A new version can be deployed and rolled back predictably.

Status: complete. A local Docker Desktop Kubernetes cluster runs the
`service-foundations` FastAPI image from a private GitHub Container Registry
package. Image pull credentials, Pod replacement, readiness, liveness,
resource requests and limits, Secret injection, failed private-image rollout,
rollback, and controlled immutable deployment are verified. GitHub Actions
validates the committed manifests, rejects committed Kubernetes Secret
resources, and publishes multi-platform service images to private GHCR
packages with immutable commit tags. Evidence and operational limits are
documented in `projects/05-reliable-deployment/REPORT.md`.

## v0.5.1 - Internal Inference Access

Expose rented GPU inference as a shared internal service without requiring
users to access the host through SSH.

Access path:

```text
Terminal or web UI -> HTTPS gateway -> vLLM on rented GPU infrastructure
```

Scope:

- HTTPS inference endpoint
- One API key per user
- API key revocation
- User identity in structured logs and metrics
- Per-user rate limits
- Streaming chat completions
- Concurrent terminal and web UI clients
- Gateway health checks and upstream failure handling
- Secrets outside the repository

SSH remains an administrator-only path for infrastructure setup and
troubleshooting. Users interact only with the authenticated inference API.
Each user receives a separate key so access can be audited, limited, and
revoked without rotating credentials for every user.

Completion criteria:

- At least five independently authenticated users can access one HTTPS
  endpoint.
- Multiple terminal or web UI clients receive streaming responses
  concurrently.
- Invalid and revoked keys are rejected without reaching vLLM.
- Rate limits and request metrics are attributed to the authenticated user.
- vLLM is not exposed directly to the public internet.
- Setup, access revocation, concurrency behavior, and operational limits are
  documented with live evidence.

Status: planned.

## v0.6 - Multi-GPU Inference

Measure distributed inference on rented multi-GPU infrastructure.

Scope:

- Tensor parallelism
- Pipeline parallelism
- NCCL
- GPU topology
- Multi-node fundamentals
- Network bottlenecks

Completion criteria:

- Single-GPU and multi-GPU results are compared.
- Scaling efficiency and bottlenecks are explained with measurements.

## v1.0 - Portfolio System

The final repository must include:

- Local Apple Silicon development
- NVIDIA deployment
- Docker Compose
- Kubernetes manifests
- Load tests
- Benchmark automation
- Dashboards
- Failure scenarios
- Quantization comparisons
- Capacity and cost reports
- Documented architectural decisions

Completion criteria:

- Every major design decision can be defended with code, tests, or measured
  evidence.
