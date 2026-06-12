# Observability

Prometheus and Grafana stack for the v0.3 production serving service.

The application exposes metrics on `GET /metrics` (see
`src/production_serving/metrics.py`). This stack scrapes that endpoint and
provides a pre-provisioned Grafana dashboard.

## Run

Start the production serving service on the host first:

```bash
SERVING_HOST=0.0.0.0 uv run production-serving
```

Binding to `0.0.0.0` makes the service reachable from Prometheus through
`host.docker.internal:8000`. Do not expose port 8000 to an untrusted network.

Then start the observability stack:

```bash
cd observability
docker compose up
```

- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (anonymous access enabled, Viewer role)

Both ports bind to localhost only.

## Contents

- `prometheus.yml` - scrape config. Scrapes `/metrics` on
  `host.docker.internal:8000` every 5 seconds.
- `compose.yaml` - Prometheus and Grafana services.
- `grafana/provisioning/datasources/datasource.yml` - provisions the
  Prometheus data source.
- `grafana/provisioning/dashboards/dashboards.yml` - provisions dashboards
  from `grafana/dashboards/`.
- `grafana/dashboards/production-serving.json` - the "Production Serving"
  dashboard.

## Dashboard panels

The "Production Serving" dashboard includes:

- **Request rate** - `sum(rate(http_requests_total[1m])) by (path, method)`
- **Error rate** - 4xx and 5xx share of total requests, 5 minute window
- **Active requests** - `http_requests_active`
- **Request duration (p50 / p95 / p99)** - quantiles of
  `http_request_duration_seconds`, by path
- **Time to first token (p50 / p95 / p99)** - quantiles of
  `chat_completion_time_to_first_token_seconds`
- **Generated token throughput** -
  `sum(rate(chat_completion_generated_tokens_total[1m]))`

## Metrics reference

| Metric | Type | Labels | Description |
| --- | --- | --- | --- |
| `http_requests_total` | counter | `method`, `path`, `status` | Total HTTP requests |
| `http_request_duration_seconds` | histogram | `method`, `path` | Request duration |
| `http_requests_active` | gauge | - | In-flight HTTP requests |
| `chat_completion_time_to_first_token_seconds` | histogram | - | Time to first generated token |
| `chat_completion_generated_tokens_total` | counter | - | Total completion tokens generated |
