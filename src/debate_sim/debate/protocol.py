from __future__ import annotations

from pathlib import Path
from typing import Dict

TACTICS = [
    "pattern-seeking",
    "distrust-authority",
    "narrative-coherence",
    "anomaly-emphasis",
    "institutional-critique",
]

PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"


def load_prompt(name: str, values: Dict[str, str]) -> str:
    template = (PROMPT_DIR / name).read_text(encoding="utf-8")
    return template.format(**values)
