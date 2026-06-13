# Internal Inference Access

This project exposes rented GPU inference through an authenticated gateway.
Users connect through terminal or web clients without SSH access to the GPU
host.

Status: complete.

## Authentication

Each user receives a separate API key. The gateway reads user-to-key mappings
from `GATEWAY_API_KEYS` as a JSON object:

```bash
export GATEWAY_API_KEYS='{"user-a":"replace-me","user-b":"replace-me"}'
```

Real keys must remain outside the repository.

An administrator can revoke or restore one user's access without rotating
other users' keys:

```bash
curl -X DELETE \
  -H "X-Admin-Key: $GATEWAY_ADMIN_API_KEY" \
  http://127.0.0.1:8080/admin/users/user-a/access

curl -X PUT \
  -H "X-Admin-Key: $GATEWAY_ADMIN_API_KEY" \
  http://127.0.0.1:8080/admin/users/user-a/access
```

Revocation state is process-local in this version. A multi-replica gateway
requires a shared credential store.

Authenticated `POST /v1/chat/completions` requests are forwarded to the
configured vLLM-compatible upstream. The user's gateway key is never sent
upstream. Streaming Server-Sent Events are relayed without application-level
buffering.

Copy `.env.example` to `.env`, replace the example values, and start the
gateway:

```bash
uv run internal-inference-access
```

The gateway binds to `127.0.0.1:8080` by default. The deployed Kubernetes
Service is private, and a Cloudflare Tunnel sidecar provides the public HTTPS
entry point at `inference.kernelgallery.com`.

## Terminal client

Users only need their own API key:

```bash
export INFERENCE_API_KEY="assigned-user-key"
uv run internal-inference-chat
```

The client connects to `https://inference.kernelgallery.com` by default and
prints streaming tokens as they arrive.

Five independently authenticated terminal users were verified concurrently
against the HTTPS endpoint. All five requests streamed successfully. Structured
results are stored in `benchmarks/five-user-streaming.json`.

Live access-control evidence is stored in `benchmarks/access-control.json`.
Revoking `user-a` changed access from `200` to `401`, restoring it returned
access to `200`, and exhausting that user's burst produced one `429` without
affecting `user-b`.

## RunPod startup

`runpod/start-vllm.sh` restores the GPU runtime after a Pod restart:

Store the script on the persistent volume and configure the RunPod container
start command:

```text
bash /workspace/vllm-bench/start-vllm.sh
```

The script exits when vLLM is already healthy, avoids starting a duplicate
process, installs the pinned runtime on ephemeral local disk when required,
and replaces itself with the GPTQ vLLM server process. Running vLLM as PID 1
keeps the container alive and sends startup logs to the RunPod console.

Rate limiting uses a separate token bucket for each authenticated user.
Prometheus metrics are available at `/metrics` and label chat request outcomes
with the configured user identity.

`GET /health/live` reports whether the gateway process is serving requests.
`GET /health/ready` returns success only when the configured vLLM upstream
health endpoint responds successfully. Readiness uses a separate two-second
timeout by default so a stalled upstream does not block health checks.

## Kubernetes

The committed manifests keep the gateway private as a ClusterIP Service.
Create runtime secrets before applying them:

```bash
kubectl create secret generic internal-inference-access-secrets \
  --from-literal=api-keys="$GATEWAY_API_KEYS" \
  --from-literal=admin-api-key="$GATEWAY_ADMIN_API_KEY" \
  --from-literal=upstream-base-url="$GATEWAY_UPSTREAM_BASE_URL"

kubectl apply -f kubernetes/
```

The deployed gateway image is stored in private GHCR with an immutable digest.
Kubernetes pulls it with a repository credential stored outside Git.

The gateway Pod runs the Cloudflare Tunnel connector as a sidecar container.
Its remotely managed hostname routes to:

```text
http://localhost:8080
```

Create the tunnel token Secret outside the repository:

```bash
kubectl create secret generic internal-inference-cloudflared \
  --from-literal=tunnel-token="$CLOUDFLARE_TUNNEL_TOKEN"
```

The connector token must never be committed. The sidecar shares the Pod
network with the gateway, so the existing Cloudflare origin remains
`localhost:8080`. Once the sidecar is healthy, the temporary local
`kubectl port-forward` and Mac connector are no longer required.
