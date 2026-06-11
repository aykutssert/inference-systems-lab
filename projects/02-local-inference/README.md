# Local Inference

The v0.2 project runs Qwen3-1.7B on Apple Silicon with MLX and llama.cpp. It
includes an OpenAI-compatible HTTP API and repeatable MLX benchmarks.

## Model

- Source: `Qwen/Qwen3-1.7B`
- MLX artifact: `mlx-community/Qwen3-1.7B-4bit`
- MLX revision: `3b1b1768f8f8cf8351c712464f906e86c2b8269e`
- Generation mode: non-thinking

## Setup

```bash
uv sync --locked
```

## Download

```bash
uv run download-model
```

The model is stored in the standard Hugging Face cache and is not committed to
the repository.

Download the pinned official GGUF artifact:

```bash
uv run download-gguf-model
```

The GGUF backend uses `Qwen3-1.7B-Q8_0.gguf` from
`Qwen/Qwen3-1.7B-GGUF` at revision
`90862c4b9d2787eaed51d12237eafdfe7c5f6077`.

The llama.cpp integration manages one long-running `llama-server` process. It
uses Metal offload, non-thinking chat templates, greedy decoding, and one
server slot. Time to first token is measured from the first streamed content
chunk. Peak memory is the highest observed resident memory of the server
process during generation.

## Runtime Comparison Decision

MLX and llama.cpp are not compared directly in v0.2. The pinned MLX artifact is
4-bit while the official GGUF artifact is Q8_0. Their current memory metrics
also differ: MLX allocator peak versus llama.cpp process RSS.

Both backends are validated independently. A direct comparison remains deferred
until equivalent quantization and a shared process-level memory metric are
available.

## Smoke Test

```bash
uv run mlx-smoke-test
```

The command prints the generated response followed by a JSON measurement
record. It measures model load time, time to first token, prompt throughput,
generation throughput, and peak MLX memory.

## Benchmark Contract

The benchmark workload contains three prompt categories:

- Short definition
- Medium explanation
- Long-context summary

Each measured run will be stored as one self-contained JSONL record containing:

- Experiment and runtime metadata
- Model repository and pinned revision
- Prompt id, category, text, and token limit
- Run index and generated response
- Latency, throughput, token count, and memory metrics

The benchmark contract is defined, tested, and validated with a full MLX run.

## Runner Rules

- Load the model once per experiment.
- Run one warm-up generation with the short prompt.
- Run each benchmark prompt three measured times.
- Use non-thinking chat templates.
- Use MLX-LM greedy decoding.
- Return benchmark records in memory.

The runner does not write files or calculate aggregate statistics.

## Result Writer

The JSONL writer:

- Writes one line per benchmark record.
- Creates parent directories when needed.
- Publishes a complete file atomically.
- Refuses to overwrite an existing experiment.
- Rejects unsafe file names and mixed experiment records.

## Run Benchmark

```bash
uv run mlx-benchmark
```

The command runs one warm-up and nine measured generations, then writes a
timestamped JSONL file to the repository `benchmarks/mlx` directory.

Use a different output directory when needed:

```bash
uv run mlx-benchmark --output-directory /tmp/mlx-results
```

## Benchmark Summary

The summary layer calculates prompt-level minimum, average, and maximum values
for:

- Time to first token
- Prompt throughput
- Generation throughput
- Peak memory

It also records run count and stable token counts. The current step implements
the aggregation model.

## Result Reader

The JSONL reader reconstructs validated benchmark records and reports malformed
data with the source file and line number. It rejects:

- Invalid JSON
- Non-object records
- Blank lines
- Missing or invalid fields
- Empty benchmark files

## Markdown Report

The renderer converts a benchmark summary into deterministic Markdown with:

- Experiment and runtime metadata
- Model load time
- One table row per prompt
- Minimum, average, and maximum metrics

Raw model responses remain in JSONL and are not duplicated in the report.

Generate a report next to a benchmark file:

```bash
uv run mlx-benchmark-report ../../benchmarks/mlx/<experiment-id>.jsonl
```

The command writes `<experiment-id>.md` atomically and refuses to overwrite an
existing report. Use `--output <path>` to select a different destination.

## OpenAI-Compatible API

Start the local MLX service:

```bash
uv run local-inference-api
```

The service listens on `http://127.0.0.1:8000` and loads the pinned model during
startup. The first API slice provides:

- `GET /v1/models`
- `POST /v1/chat/completions`
- `GET /health/live`
- `GET /health/ready`
- Developer, system, user, and assistant text messages
- `max_completion_tokens` and deprecated `max_tokens`
- Token usage in completion responses
- Server-Sent Events streaming with final token usage

Multiple choices, tools, and sampling controls are not supported in this
slice. Unsupported request fields are rejected instead of being ignored.
Backend generation failures return an OpenAI-shaped `503` response without
exposing internal exception details.

The compatibility contract is tested with the official OpenAI Python SDK:

```python
from openai import OpenAI

client = OpenAI(
    api_key="local",
    base_url="http://127.0.0.1:8000/v1",
)
response = client.chat.completions.create(
    model="mlx-community/Qwen3-1.7B-4bit",
    messages=[{"role": "user", "content": "Explain inference latency."}],
)
```
