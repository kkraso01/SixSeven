from __future__ import annotations

import json
from pathlib import Path

from ..schemas import MemoryState


def save_memory(path: Path, memory: MemoryState) -> None:
    path.write_text(json.dumps(memory.model_dump(), indent=2))
