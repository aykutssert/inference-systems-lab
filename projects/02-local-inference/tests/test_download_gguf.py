from pathlib import Path
from typing import Any

from local_inference.config import (
    GGUF_MODEL_FILE,
    GGUF_MODEL_ID,
    GGUF_MODEL_REVISION,
)
from local_inference.download_gguf import download_model


def test_download_model_uses_pinned_gguf_artifact(tmp_path: Path) -> None:
    expected_path = tmp_path / GGUF_MODEL_FILE
    received: dict[str, Any] = {}

    def fake_download(**kwargs: Any) -> str:
        received.update(kwargs)
        return str(expected_path)

    result = download_model(fake_download)

    assert result == expected_path
    assert received == {
        "repo_id": GGUF_MODEL_ID,
        "filename": GGUF_MODEL_FILE,
        "revision": GGUF_MODEL_REVISION,
    }
