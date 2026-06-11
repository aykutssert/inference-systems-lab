from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationChunk:
    text: str
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    generation_tokens: int | None = None

    @property
    def is_final(self) -> bool:
        return self.prompt_tokens is not None and self.generation_tokens is not None
