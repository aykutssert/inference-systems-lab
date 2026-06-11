import json
import time
from collections.abc import Callable
from pathlib import Path

from huggingface_hub import hf_hub_download

from local_inference.config import (
    GGUF_MODEL_FILE,
    GGUF_MODEL_ID,
    GGUF_MODEL_REVISION,
)

DownloadFunction = Callable[..., str]


def download_model(download: DownloadFunction = hf_hub_download) -> Path:
    return Path(
        download(
            repo_id=GGUF_MODEL_ID,
            filename=GGUF_MODEL_FILE,
            revision=GGUF_MODEL_REVISION,
        )
    )


def main() -> None:
    started_at = time.perf_counter()
    model_path = download_model()
    elapsed_seconds = time.perf_counter() - started_at

    print(
        json.dumps(
            {
                "model": GGUF_MODEL_ID,
                "revision": GGUF_MODEL_REVISION,
                "file": GGUF_MODEL_FILE,
                "path": str(model_path),
                "download_seconds": round(elapsed_seconds, 3),
            },
            indent=2,
        )
    )
