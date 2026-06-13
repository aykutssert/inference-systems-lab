# NVIDIA Inference Report

## Environment

- Provider: RunPod rented GPU instance
- GPU: NVIDIA RTX PRO 4000 Blackwell
- GPU memory: 24,467 MiB
- Client path: local benchmark runner through an SSH port-forwarding tunnel
- vLLM: 0.22.1
- PyTorch: 2.11.0+cu128
- Model: `Qwen/Qwen3-1.7B`
- Precision: FP16
- Context window: 40,960 tokens
- KV cache: 17.16 GiB and 160,624 tokens
- Initial server startup: approximately 11 minutes

The Python environment was stored on network-mounted storage. Package imports,
model registry inspection, kernel compilation, CUDA graph capture, and
FlashInfer tuning dominated the initial startup. Startup time is therefore
reported separately from request latency.

The rented GPU is intentionally part of the system design. Local Apple Silicon
is used for development and CPU-independent tooling, while CUDA, vLLM,
continuous batching, KV-cache allocation, quantization, and out-of-memory
experiments run on disposable RunPod infrastructure.

The FlashInfer sampler included in the installed vLLM environment rejected the
Blackwell `sm_120` capability during engine profiling. Setting
`VLLM_USE_FLASHINFER_SAMPLER=0` selects vLLM's PyTorch-native sampler while
leaving model execution and quantization on the GPU.

## Method

Each steady-state run sent 8 identical chat requests and generated 128 tokens
per request. The benchmark records timestamps while consuming the SSE stream.
TTFT is measured from request start to the first content chunk. TPOT is the
interval from the first to the last content chunk divided by the remaining
output-token count.

## FP16 Baseline

| Concurrency | Success | Output tok/s | TTFT mean | TPOT mean | Latency p95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8/8 | 114.96 | 244 ms | 6.84 ms | 1.127 s |
| 2 | 8/8 | 217.48 | 227 ms | 7.47 ms | 1.192 s |
| 4 | 8/8 | 429.06 | 219 ms | 7.30 ms | 1.195 s |
| 8 warm | 8/8 | 849.13 | 215 ms | 7.17 ms | 1.203 s |

Throughput scaled almost linearly from concurrency 1 through 8 while warm
per-request latency remained nearly flat. This is evidence that vLLM
continuous batching combined these short requests efficiently on this GPU.

The first concurrency-8 run was an outlier: 137.29 output tokens/s, 1.78 s
mean TTFT, and 7.44 s p95 latency. Repeating the same run immediately produced
the warm result above. The cold result is retained as
`benchmarks/fp16-c8.json` because it shows that a previously unseen batch shape
can incur a one-time compile or tuning penalty. Capacity conclusions use the
warm result.

## GPTQ Int8 Baseline

| Concurrency | Success | Output tok/s | TTFT mean | TPOT mean | Latency p95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8/8 | 152.61 | 233 ms | 4.77 ms | 0.849 s |
| 2 | 8/8 | 310.82 | 176 ms | 4.88 ms | 0.850 s |
| 4 | 8/8 | 599.40 | 204 ms | 4.89 ms | 0.876 s |
| 8 | 8/8 | 1,168.70 | 290 ms | 4.46 ms | 0.871 s |

The first GPTQ runs at new batch shapes also incurred compile or tuning
penalties. The comparison below uses isolated warm runs. Benchmark processes
were not run in parallel because sharing the GPU would invalidate latency and
throughput results.

## FP16 vs GPTQ Int8

| Concurrency | FP16 tok/s | GPTQ tok/s | Throughput gain | FP16 p95 | GPTQ p95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 114.96 | 152.61 | 32.8% | 1.127 s | 0.849 s |
| 2 | 217.48 | 310.82 | 42.9% | 1.192 s | 0.850 s |
| 4 | 429.06 | 599.40 | 39.7% | 1.195 s | 0.876 s |
| 8 | 849.13 | 1,168.70 | 37.6% | 1.203 s | 0.871 s |

GPTQ Int8 delivered 32.8% to 42.9% more output throughput and lower p95
latency at every tested concurrency. Its model weights used 1.92 GiB instead
of FP16's 3.22 GiB, a 40.4% reduction.

| Metric | FP16 | GPTQ Int8 |
| --- | ---: | ---: |
| Model weight memory | 3.22 GiB | 1.92 GiB |
| Available KV cache | 17.16 GiB | 18.18 GiB |
| KV cache tokens | 160,624 | 170,160 |
| Full-context concurrency estimate | 3.92x | 4.15x |
| Idle GPU memory reported by `nvidia-smi` | 22,102 MiB | 21,924 MiB |

Quantization did not reduce total allocated GPU memory proportionally because
vLLM was configured with `--gpu-memory-utilization 0.9`. vLLM converted most
of the freed model-weight memory into additional KV cache capacity. The useful
effect is therefore higher throughput and more cache capacity, not a large
drop in the idle `nvidia-smi` number.

GPTQ Int8 is the preferred serving format for this workload on the tested
RunPod GPU. FP16 remains the unquantized reference.

## Concurrency Capacity

| Concurrency | Success | Output tok/s | TTFT mean | TPOT mean | Latency p95 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 16/16 | 2,229 | 270 ms | 4.97 ms | 0.914 s |
| 32 | 32/32 | 4,039 | 256 ms | 5.50 ms | 0.984 s |
| 64 | 64/64 | 6,405 | 303 ms | 7.22 ms | 1.235 s |
| 128 | 128/128 | 8,460 | 422 ms | 11.24 ms | 1.920 s |

All 240 requests completed without rejection or out-of-memory failure. vLLM
continued increasing aggregate throughput through concurrency 128, but the
latency cost became visible after concurrency 32. TPOT increased by 31% at
concurrency 64 and 104% at concurrency 128 relative to concurrency 32.

Concurrency 32 is the balanced operating point for this short-prompt,
128-output-token workload. Higher concurrency remains useful for offline
throughput workloads, but it degrades interactive latency.

## Controlled CUDA OOM

The running GPTQ server occupied 21.40 GiB and left 1.79 GiB free. A separate
short-lived CUDA process attempted to allocate 2.29 GiB, 512 MiB more than the
reported free memory. PyTorch raised `torch.OutOfMemoryError` and the test
process exited with code 1.

The OOM remained isolated to the allocating process:

- The vLLM health endpoint remained available.
- Model discovery still reported the GPTQ model and 40,960-token context.
- A completion request succeeded immediately after the failure.
- GPU memory returned to the pre-test 21,924 MiB allocation.

This demonstrates process-level failure isolation. A production deployment
still needs an orchestrator to restart the serving process if the OOM occurs
inside vLLM itself.

## Excluded Variant

AWQ was removed from the comparison. The available Qwen3-1.7B AWQ checkpoint
is community-maintained under GPL-3.0, while the GPTQ Int8 checkpoint is
published by Qwen under Apache-2.0. The AWQ startup also exposed a
Blackwell-FlashInfer compatibility failure. Comparing the official FP16 and
GPTQ artifacts keeps the experiment reproducible and focused on a defensible
production choice.
