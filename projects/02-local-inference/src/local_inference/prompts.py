from local_inference.benchmark_schema import BenchmarkPrompt

LONG_CONTEXT_BLOCK = (
    "The service accepts requests with different prompt and output lengths. "
    "Prefill processes prompt tokens in parallel, while decode produces output "
    "tokens sequentially. Continuous batching combines active requests to improve "
    "device utilization. The scheduler must still protect latency because queued "
    "requests wait before inference begins. Each active sequence consumes KV cache "
    "memory, so context length and concurrency compete for limited capacity. When "
    "memory pressure rises, the server may delay, preempt, or reject work. Operators "
    "therefore compare time to first token, time per output token, total throughput, "
    "memory use, and tail latency instead of relying on a single speed metric."
)

LONG_CONTEXT = "\n\n".join(
    f"Observation {index}: {LONG_CONTEXT_BLOCK}" for index in range(1, 9)
)

BENCHMARK_PROMPTS = (
    BenchmarkPrompt(
        id="short-definition",
        category="short",
        text="Define model inference latency in one short sentence.",
        max_tokens=64,
    ),
    BenchmarkPrompt(
        id="medium-explanation",
        category="medium",
        text=(
            "Explain how batching affects latency and throughput in an inference "
            "service. Use one short paragraph."
        ),
        max_tokens=160,
    ),
    BenchmarkPrompt(
        id="long-context-summary",
        category="long-context",
        text=(
            f"{LONG_CONTEXT}\n\n"
            "Summarize the main latency, throughput, and memory tradeoffs in two "
            "sentences."
        ),
        max_tokens=128,
    ),
)
