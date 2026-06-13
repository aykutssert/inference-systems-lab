import argparse
from dataclasses import dataclass

# Verified via transformers.AutoConfig.from_pretrained: Qwen/Qwen3-1.7B,
# Qwen/Qwen3-1.7B-GPTQ-Int8, and Orion-zhen/Qwen3-1.7B-AWQ all report
# max_position_embeddings=40960, matching the pinned
# mlx-community/Qwen3-1.7B-4bit checkpoint used in v0.2 and v0.3.
DEFAULT_CONTEXT_WINDOW = 40960
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
DEFAULT_GPU_MEMORY_UTILIZATION = 0.9


@dataclass(frozen=True)
class ModelVariant:
    name: str
    model_id: str
    quantization: str | None
    dtype: str
    context_window: int
    notes: str


MODEL_VARIANTS: dict[str, ModelVariant] = {
    "fp16": ModelVariant(
        name="fp16",
        model_id="Qwen/Qwen3-1.7B",
        quantization=None,
        dtype="float16",
        context_window=DEFAULT_CONTEXT_WINDOW,
        notes=(
            "Unquantized baseline. Largest VRAM footprint and a reference "
            "point for accuracy."
        ),
    ),
    "gptq": ModelVariant(
        name="gptq",
        model_id="Qwen/Qwen3-1.7B-GPTQ-Int8",
        quantization="gptq",
        dtype="auto",
        context_window=DEFAULT_CONTEXT_WINDOW,
        notes="Official Qwen 8-bit GPTQ checkpoint.",
    ),
}


def build_vllm_serve_command(
    variant: ModelVariant,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    served_model_name: str | None = None,
    max_model_len: int | None = None,
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION,
) -> list[str]:
    command = [
        "vllm",
        "serve",
        variant.model_id,
        "--host",
        host,
        "--port",
        str(port),
        "--dtype",
        variant.dtype,
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
    ]
    if variant.quantization is not None:
        command += ["--quantization", variant.quantization]
    if served_model_name is not None:
        command += ["--served-model-name", served_model_name]
    if max_model_len is not None:
        command += ["--max-model-len", str(max_model_len)]
    return command


def print_serve_command() -> None:
    parser = argparse.ArgumentParser(
        description="Print a vllm serve command for one of the prepared model variants"
    )
    parser.add_argument("variant", choices=sorted(MODEL_VARIANTS))
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=DEFAULT_GPU_MEMORY_UTILIZATION
    )
    args = parser.parse_args()

    command = build_vllm_serve_command(
        MODEL_VARIANTS[args.variant],
        host=args.host,
        port=args.port,
        served_model_name=args.served_model_name,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print(" ".join(command))


if __name__ == "__main__":  # pragma: no cover
    print_serve_command()
