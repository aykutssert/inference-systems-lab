from nvidia_inference.config import MODEL_VARIANTS, build_vllm_serve_command


def test_model_variants_cover_fp16_and_gptq() -> None:
    assert set(MODEL_VARIANTS) == {"fp16", "gptq"}
    for variant in MODEL_VARIANTS.values():
        assert variant.model_id
        assert variant.context_window > 0


def test_fp16_command_has_no_quantization_flag() -> None:
    command = build_vllm_serve_command(MODEL_VARIANTS["fp16"])

    assert command[:3] == ["vllm", "serve", "Qwen/Qwen3-1.7B"]
    assert "--quantization" not in command
    assert "--dtype" in command
    assert command[command.index("--dtype") + 1] == "float16"


def test_gptq_command_includes_quantization_flag() -> None:
    command = build_vllm_serve_command(MODEL_VARIANTS["gptq"])

    assert "--quantization" in command
    assert command[command.index("--quantization") + 1] == "gptq"


def test_command_includes_optional_overrides() -> None:
    command = build_vllm_serve_command(
        MODEL_VARIANTS["fp16"],
        host="127.0.0.1",
        port=9000,
        served_model_name="qwen3-1.7b-fp16",
        max_model_len=8192,
        gpu_memory_utilization=0.8,
    )

    assert "--host" in command
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert "--port" in command
    assert command[command.index("--port") + 1] == "9000"
    assert "--served-model-name" in command
    assert command[command.index("--served-model-name") + 1] == "qwen3-1.7b-fp16"
    assert "--max-model-len" in command
    assert command[command.index("--max-model-len") + 1] == "8192"
    assert "--gpu-memory-utilization" in command
    assert command[command.index("--gpu-memory-utilization") + 1] == "0.8"


def test_command_omits_optional_flags_by_default() -> None:
    command = build_vllm_serve_command(MODEL_VARIANTS["fp16"])

    assert "--served-model-name" not in command
    assert "--max-model-len" not in command
