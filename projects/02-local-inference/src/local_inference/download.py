import json
import time
from pathlib import Path

from huggingface_hub import snapshot_download

from local_inference.config import MODEL_ID, MODEL_REVISION


def main() -> None:
    started_at = time.perf_counter()
    model_path = Path(snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION))
    elapsed_seconds = time.perf_counter() - started_at

    print(
        json.dumps(
            {
                "model": MODEL_ID,
                "revision": MODEL_REVISION,
                "path": str(model_path),
                "download_seconds": round(elapsed_seconds, 3),
            },
            indent=2,
        )
    )
