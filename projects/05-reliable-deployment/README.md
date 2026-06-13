# Reliable Deployment

The v0.5 project turns the inference workload into a recoverable,
version-controlled deployment.

Status: in progress. The `service-foundations` FastAPI image runs from a
private GitHub Container Registry package on the local Docker Desktop
Kubernetes cluster.

## First deployment

Apply the Deployment:

```bash
kubectl apply -f kubernetes/
```

Inspect the desired state and the Pod created by the Deployment:

```bash
kubectl get deployments
kubectl get pods
kubectl get services
```

The Deployment maintains one replica. Deleting its Pod should cause
Kubernetes to create a replacement automatically.

The Service selects Pods labeled `app: service-foundations` and gives them a
stable cluster address. Its default `ClusterIP` type is reachable only from
inside the cluster.

The readiness probe checks `/health/ready` every five seconds. Kubernetes
keeps a Pod out of Service endpoints until the check succeeds.

The liveness probe checks `/health/live` every ten seconds. Three consecutive
failures cause kubelet to restart the container.

Resource requests reserve scheduling capacity. Resource limits cap container
usage. The FastAPI workload requests 50 millicores and 64 MiB, with limits of
500 millicores and 256 MiB.

## Local secret

Create the Secret before applying the Deployment:

```bash
kubectl create secret generic service-foundations-secrets \
  --from-literal=internal-api-token="$INTERNAL_API_TOKEN"
```

The Deployment reads the value into `INTERNAL_API_TOKEN` through a
`secretKeyRef`. Only `.env.example` is committed. Real values must remain
outside the repository.

The application image is stored as a private package in GitHub Container
Registry. Create a local pull credential before applying the Deployment:

```bash
kubectl create secret docker-registry ghcr-credentials \
  --docker-server=ghcr.io \
  --docker-username=aykutssert \
  --docker-password="$(gh auth token)"
```

The Deployment references `ghcr-credentials` through `imagePullSecrets`.
Registry credentials remain in the cluster and are never committed.

## Validation

GitHub Actions installs the same `kubectl` minor version as the local cluster
and renders every committed manifest with client-side dry-run. CI also rejects
committed Kubernetes `Secret` manifests. Local validation additionally uses
the live API server:

```bash
kubectl apply --dry-run=server -f kubernetes/ -o name
```

Pushes to `main` build the service image for `linux/amd64` and `linux/arm64`,
then publish it to the private GHCR package with an immutable
`sha-<commit>` tag. The workflow does not overwrite the release tag or deploy
to a cluster automatically.

Verified failure behavior:

- Deleting the active FastAPI Pod caused the Deployment to create a
  replacement from the same GHCR image digest.
- A nonexistent private image tag failed with `ErrImagePull`.
- The old ready FastAPI Pod remained available during the failed rollout.
- `kubectl rollout undo` restored the pinned `0.1.0` image.
- Service liveness and readiness endpoints returned successful responses after
  rollback.
