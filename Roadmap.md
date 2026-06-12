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

## v0.4 - NVIDIA Inference

Move the same workload to a rented NVIDIA GPU environment.

Scope:

- CUDA and VRAM behavior
- vLLM
- KV cache
- Continuous batching
- FP16, AWQ, and GPTQ
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
