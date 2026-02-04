from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    value = os.getenv(key)
    return int(value) if value is not None else default


def _env_float(key: str, default: float) -> float:
    value = os.getenv(key)
    return float(value) if value is not None else default


def _env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class DebateConfig:
    base_url: str = "http://localhost:11434"
    api_mode: str = "ollama"  # "ollama" or "openai"
    moderator_model: str = "llama3.1:8b"
    conspiracy_model: str = "llama3.1:8b"
    scientific_model: str = "llama3.1:8b"
    moderator_temperature: float = 0.2
    conspiracy_temperature: float = 0.6
    scientific_temperature: float = 0.2
    max_tokens: int = 600
    rounds: int = 3
    word_limit: int = 180
    seed: Optional[int] = None
    output_dir: str = "artifacts"
    run_analysis: bool = True
    analysis_shift_threshold: int = 5
    analysis_similarity_method: str = "tfidf"

    @classmethod
    def from_env(cls) -> "DebateConfig":
        return cls(
            base_url=_env("DEBATE_BASE_URL", cls.base_url),
            api_mode=_env("DEBATE_API_MODE", cls.api_mode),
            moderator_model=_env("DEBATE_MODERATOR_MODEL", cls.moderator_model),
            conspiracy_model=_env("DEBATE_CONSPIRACY_MODEL", cls.conspiracy_model),
            scientific_model=_env("DEBATE_SCIENTIFIC_MODEL", cls.scientific_model),
            moderator_temperature=_env_float(
                "DEBATE_MODERATOR_TEMPERATURE", cls.moderator_temperature
            ),
            conspiracy_temperature=_env_float(
                "DEBATE_CONSPIRACY_TEMPERATURE", cls.conspiracy_temperature
            ),
            scientific_temperature=_env_float(
                "DEBATE_SCIENTIFIC_TEMPERATURE", cls.scientific_temperature
            ),
            max_tokens=_env_int("DEBATE_MAX_TOKENS", cls.max_tokens),
            rounds=_env_int("DEBATE_ROUNDS", cls.rounds),
            word_limit=_env_int("DEBATE_WORD_LIMIT", cls.word_limit),
            seed=int(os.getenv("DEBATE_SEED")) if os.getenv("DEBATE_SEED") else None,
            output_dir=_env("DEBATE_OUTPUT_DIR", cls.output_dir),
            run_analysis=_env_bool("DEBATE_RUN_ANALYSIS", cls.run_analysis),
            analysis_shift_threshold=_env_int(
                "DEBATE_ANALYSIS_SHIFT_THRESHOLD", cls.analysis_shift_threshold
            ),
            analysis_similarity_method=_env(
                "DEBATE_ANALYSIS_SIMILARITY_METHOD", cls.analysis_similarity_method
            ),
        )
