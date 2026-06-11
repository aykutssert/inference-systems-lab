# Service Foundations

The v0.1 project establishes the service lifecycle and development baseline
used by later inference projects. It intentionally has no model dependency.

## Requirements

- Python 3.13
- uv 0.11.20 or newer
- Docker with Compose support

## Run Locally

```bash
uv sync --locked
uv run service-foundations
```

The service listens on `http://127.0.0.1:8000` by default.

```bash
curl http://127.0.0.1:8000/health/live
curl http://127.0.0.1:8000/health/ready
```

## Run With Docker

```bash
docker compose up --build
```

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `APP_SERVICE_NAME` | `service-foundations` | Service name used in metadata and logs |
| `APP_SERVICE_VERSION` | `0.1.0` | Reported service version |
| `APP_ENVIRONMENT` | `development` | Runtime environment |
| `APP_HOST` | `127.0.0.1` | Bind host |
| `APP_PORT` | `8000` | Bind port |
| `APP_LOG_LEVEL` | `info` | Python logging level |

## Quality Checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
```
