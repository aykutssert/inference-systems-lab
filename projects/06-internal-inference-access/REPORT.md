# Internal Inference Access Report

## Result

Project 06 turned the rented RunPod GPU from an administrator-operated host
into a shared internal inference service. Users access one HTTPS endpoint with
individual credentials and do not receive SSH access.

The verified request path is:

```text
Terminal client
  -> inference.kernelgallery.com
  -> Cloudflare Tunnel
  -> Kubernetes FastAPI gateway
  -> RunPod vLLM
  -> Qwen3-1.7B GPTQ on an RTX PRO 4000
```

## Live Verification

- Five independently authenticated users streamed responses concurrently.
- All five requests succeeded in 3.73 seconds of wall time.
- Measured time to first token ranged from 1.83 to 3.71 seconds.
- Revoking one user changed that user's response from HTTP 200 to HTTP 401.
- Restoring the user returned access to HTTP 200.
- One user exhausted a burst allowance and received HTTP 429 while another
  user continued to receive HTTP 200.
- Prometheus counters attributed successful and rate-limited requests to the
  authenticated user.
- Gateway liveness and upstream readiness returned HTTP 200 through the
  public HTTPS endpoint.

Structured results are stored in `benchmarks/five-user-streaming.json` and
`benchmarks/access-control.json`.

## Restart Recovery

RunPod container storage is ephemeral, while `/workspace` is persistent. The
startup script is stored on `/workspace` and configured as the container start
command. After a Pod reset it:

1. Checks whether vLLM is already healthy.
2. Installs the pinned vLLM 0.22.1 runtime on local ephemeral storage.
3. Configures the required CUDA runtime library path.
4. Starts `Qwen/Qwen3-1.7B-GPTQ-Int8` with a 40,960-token context limit.
5. Runs vLLM as the container's main process.

After the reset, both the RunPod health endpoint and the gateway readiness
endpoint recovered without an interactive SSH setup.

## Security Boundary

- End users receive only a gateway API key.
- The RunPod SSH endpoint remains administrator-only.
- Gateway, tunnel, registry, and upstream credentials are Kubernetes Secrets
  or ignored environment files.
- User credentials are consumed by the gateway and are not forwarded to
  vLLM.
- vLLM is reached through the configured upstream URL rather than exposed as
  the user-facing service.
- The Kubernetes gateway Service is ClusterIP and the Cloudflare connector
  runs in the same Pod.

## Operational Limits

- Revocation state is process-local and would need shared storage before
  running multiple gateway replicas.
- The single rented GPU remains the inference capacity boundary.
- RunPod restart time includes runtime installation and model loading.
- Cloudflare Tunnel provides transport exposure, not application
  authorization. The gateway API key remains required.
- API keys are static secrets in this version. Production credential
  lifecycle management would require an identity provider or secret manager.
