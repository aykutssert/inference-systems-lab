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

## Real MLX Verification

The pinned `mlx-community/Qwen3-1.7B-4bit` model was verified through the
running v0.3 service on June 12, 2026:

- First content chunk: 1.296 seconds
- Content chunks: 24
- Completion tokens: 24
- Final order: finish reason, usage, `[DONE]`
- Non-streaming completion remained compatible

Disconnect cancellation remains a separate task.
