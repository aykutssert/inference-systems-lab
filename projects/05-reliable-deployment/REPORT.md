# Reliable Deployment Report

## Outcome

v0.5 packages the service as an immutable artifact and operates it as a
recoverable Kubernetes workload.

Completed controls:

- Private image storage in GitHub Container Registry
- Immutable commit-based image tags
- Multi-platform image builds for `linux/amd64` and `linux/arm64`
- Kubernetes Deployment and stable ClusterIP Service
- Readiness and liveness probes
- CPU and memory requests and limits
- Secret injection without committed credentials
- Automatic Pod replacement
- Controlled rollout and rollback
- Server-side manifest validation in CI

## Environment

- Cluster: local Docker Desktop Kubernetes
- Provisioner: kind
- Kubernetes: 1.34.3
- Nodes: one
- Application: `service-foundations` FastAPI service
- Registry: private GitHub Container Registry package
- Image policy: immutable `sha-<commit>` tags

The local cluster is appropriate for validating orchestration behavior without
renting a GPU. NVIDIA inference remains a separate workload from v0.4 and can
later use the same deployment controls on GPU infrastructure.

## Deployment Model

GitHub Actions builds the service image on each push to `main` and publishes
both AMD64 and ARM64 variants. The workflow does not deploy automatically.

The committed Kubernetes Deployment pins a reviewed image tag:

```text
sha-9e9c0f2adf7be566d17ed00848c5ad3ace59c311
```

This separates artifact creation from deployment approval. A release consists
of reviewing the new immutable tag, updating the manifest, applying it, and
waiting for Kubernetes rollout status.

## Runtime Controls

| Control | Configuration |
| --- | --- |
| Replicas | 1 |
| Readiness | `GET /health/ready` every 5 seconds |
| Liveness | `GET /health/live` every 10 seconds |
| CPU request | 50 millicores |
| CPU limit | 500 millicores |
| Memory request | 64 MiB |
| Memory limit | 256 MiB |
| Service | ClusterIP port 80 to container port 8000 |

The readiness probe prevents an unhealthy Pod from receiving Service traffic.
The liveness probe allows Kubernetes to restart an unhealthy container.
Resource requests guide scheduling, while limits bound resource consumption.

## Secrets And Access

The private registry credential is stored in the cluster as
`ghcr-credentials`. The application token is stored as
`service-foundations-secrets` and injected through `secretKeyRef`.

No Kubernetes Secret manifest or real credential is committed. CI rejects
committed YAML resources with `kind: Secret`.

## Recovery Tests

Automatic recovery was verified by deleting the active Pod. The Deployment
created a replacement using the same private image digest, and the replacement
became ready.

A failed rollout was tested with a nonexistent private image tag:

- The new Pod entered `ErrImagePull`.
- The existing ready Pod remained available.
- `kubectl rollout undo` restored the previous image.
- Liveness and readiness endpoints succeeded after rollback.

A controlled deployment of the immutable CI image also completed successfully.
The replacement Pod reached `1/1 Ready`, and the Service returned:

```text
GET /health/live  -> {"status":"ok"}
GET /health/ready -> {"status":"ready"}
```

The running container resolved to OCI index digest:

```text
sha256:008365f661bcd7116aa89a5a5f103cbf11efa694da471bca239e254f334332b7
```

## Continuous Integration

The final v0.5 workflow passed all jobs:

- Service formatting, linting, type checking, tests, and container build
- Kubernetes server-side validation against a temporary kind cluster
- Secret-manifest policy check
- Multi-platform private image publication

GitHub Actions run `27469060997` completed successfully on June 13, 2026.

## Operational Limits

- The cluster has one node, so it validates process and Pod recovery but not
  node failure or high availability.
- One replica avoids duplicate local resource use but causes brief reduced
  capacity during replacement.
- Registry and application Secrets are created manually in each cluster.
- Deployment approval is represented by a reviewed manifest change, not a
  dedicated production release environment.
- GPU scheduling and model artifact mounting are deferred to later
  infrastructure work.

## Completion

Failed instances recover automatically. New immutable versions can be deployed
and rolled back predictably. The v0.5 completion criteria are satisfied.
