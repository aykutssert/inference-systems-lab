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
- Compare MLX and llama.cpp using the same model family and workload.

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

## v0.3 - Production Serving

Add the service controls required under concurrent load.

Scope:

- Streaming
- Request validation
- Timeouts and cancellation
- Rate limiting
- Queueing and backpressure
- Metrics and dashboards
- Load and failure testing

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
