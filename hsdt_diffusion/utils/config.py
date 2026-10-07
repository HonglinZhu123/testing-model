from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    for section in ("model", "diffusion", "data", "train"):
        if section not in config:
            raise KeyError(f"missing config section: {section}")
    return config
