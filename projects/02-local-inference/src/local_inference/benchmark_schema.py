import json
from dataclasses import asdict, dataclass
from typing import Literal


@dataclass(frozen=True)
class BenchmarkPrompt:
    id: str
    category: Literal["short", "medium", "long-context"]
    text: str
    max_tokens: int

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Prompt id must not be empty")
        if not self.text.strip():
            raise ValueError("Prompt text must not be empty")
        if self.max_tokens < 1:
            raise ValueError("Prompt max_tokens must be positive")


@dataclass(frozen=True)
class ExperimentMetadata:
    schema_version: str
    experiment_id: str
    recorded_at: str
    backend: Literal["mlx", "llama.cpp"]
    model: str
    model_revision: str
    generation_mode: Literal["non-thinking"]
    warmup_runs: int
    measured_runs: int
    python_version: str
    platform: str
    architecture: str
    runtime_version: str

    def __post_init__(self) -> None:
        if not self.experiment_id.strip():
            raise ValueError("Experiment id must not be empty")
        if self.warmup_runs < 0:
            raise ValueError("Warmup runs must not be negative")
        if self.measured_runs < 1:
            raise ValueError("Measured runs must be positive")


@dataclass(frozen=True)
class RunMetrics:
    load_seconds: float
    time_to_first_token_seconds: float
    prompt_tokens: int
    prompt_tokens_per_second: float
    generation_tokens: int
    generation_tokens_per_second: float
    peak_memory_gb: float

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(value < 0 for value in values.values()):
            raise ValueError("Run metrics must not be negative")


@dataclass(frozen=True)
class BenchmarkRecord:
    experiment: ExperimentMetadata
    prompt: BenchmarkPrompt
    run_index: int
    response: str
    finish_reason: str | None
    metrics: RunMetrics

    def __post_init__(self) -> None:
        if self.run_index < 1:
            raise ValueError("Run index must be positive")
        if not self.response.strip():
            raise ValueError("Response must not be empty")

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), sort_keys=True)
