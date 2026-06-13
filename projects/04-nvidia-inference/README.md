# NVIDIA Inference

The v0.4 project moves the v0.2/v0.3 workload from local Apple Silicon (MLX)
to a rented RunPod NVIDIA GPU running vLLM. Benchmarks run from the local
development machine through an SSH tunnel to the remote OpenAI-compatible
endpoint.

Status: complete. vLLM 0.22.1 was evaluated on a rented RunPod instance with
an NVIDIA RTX PRO 4000 Blackwell 24 GB GPU. Health, model discovery,
non-streaming, streaming, quantized inference, concurrency capacity, and
controlled CUDA out-of-memory isolation were verified.

## Model variants

`Qwen/Qwen3-1.7B` is the NVIDIA-side equivalent of the
`mlx-community/Qwen3-1.7B-4bit` model used in v0.2 and v0.3. Verified via
`transformers.AutoConfig.from_pretrained`, both variants below report
`max_position_embeddings=40960`, matching the pinned MLX checkpoint.

| Variant | Model ID | Quantization | Notes |
| --- | --- | --- | --- |
| fp16 | `Qwen/Qwen3-1.7B` | none | Unquantized baseline, largest VRAM use |
| gptq | `Qwen/Qwen3-1.7B-GPTQ-Int8` | gptq | Official Qwen 8-bit GPTQ checkpoint |

## Generating a vllm serve command

```bash
uv run nvidia-serve-command fp16
uv run nvidia-serve-command gptq --served-model-name qwen3-1.7b-gptq
```

Each invocation prints a `vllm serve ...` command for the chosen variant.
Run the printed command on the rented GPU instance once vLLM and CUDA are
installed there.

The RunPod Blackwell environment requires the PyTorch-native sampler because
the FlashInfer sampler bundled with this vLLM installation rejects the
`sm_120` device capability during engine profiling:

```bash
export VLLM_USE_FLASHINFER_SAMPLER=0
```

## Benchmarking

The v0.4 benchmark records per-request TTFT, TPOT, total latency, token counts,
and aggregate p50, p95, p99, request throughput, and output-token throughput:

```bash
uv run nvidia-benchmark \
  --model Qwen/Qwen3-1.7B \
  --requests 8 \
  --concurrency 1 \
  --output benchmarks/fp16-c1.json
```

The FP16 and GPTQ baselines use 8 requests with 128 generated tokens at
concurrency `1`, `2`, `4`, and `8`. Startup and first compile time are recorded
separately and excluded from steady-state latency. GPTQ delivered 32.8% to
42.9% higher output throughput in these tests. See [REPORT.md](REPORT.md) for
results and analysis.

## Verified FP16 environment

- Provider: RunPod rented GPU instance
- GPU: NVIDIA RTX PRO 4000 Blackwell
- Access: SSH with local port forwarding to the remote vLLM endpoint
- vLLM: 0.22.1
- PyTorch: 2.11.0+cu128
- Model: `Qwen/Qwen3-1.7B`
- Model context window: 40,960 tokens
- Loaded model weights: 3.22 GiB
- KV cache: 17.16 GiB, 160,624 tokens
- Initial idle GPU memory: 22,102 / 24,467 MiB
- Initial server startup: approximately 11 minutes on network-mounted storage

The measured balanced operating point is concurrency 32 for the tested
interactive workload. Higher concurrency increases aggregate throughput but
degrades per-token and tail latency. A controlled CUDA OOM in a separate
process did not interrupt the running vLLM service.
