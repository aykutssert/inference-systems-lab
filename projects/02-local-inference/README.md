# Local Inference

The v0.2 project runs Qwen3-1.7B on Apple Silicon with MLX before adding an
HTTP API or a second runtime.

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

## Smoke Test

```bash
uv run mlx-smoke-test
```

The command prints the generated response followed by a JSON measurement
record. It measures model load time, time to first token, prompt throughput,
generation throughput, and peak MLX memory.
