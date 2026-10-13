import json
import os
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("ISL_DATA_DIR", PROJECT_ROOT / "data")).expanduser().resolve()
STATIC_DATA_ROOT = Path(
    os.environ.get("ISL_STATIC_DATA_DIR", DATA_ROOT / "raw" / "islrtc-referred")
).expanduser().resolve()
ARTIFACT_ROOT = Path(
    os.environ.get("ISL_ARTIFACT_DIR", PROJECT_ROOT / "artifacts")
).expanduser().resolve()
CONFIG_PATH = Path(os.environ.get("ISL_CONFIG", PROJECT_ROOT / "config" / "default.json")).expanduser().resolve()


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as config_file:
        return json.load(config_file)
