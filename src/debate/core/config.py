from __future__ import annotations

import os
from configparser import ConfigParser
from dataclasses import dataclass, field
from pathlib import Path

from debate.analysis.lexicons import (
    MODALITY_STRONG_WORDS,
    MODALITY_WEAK_WORDS,
    UNCERTAINTY_WORDS,
)
from debate.core.model_pool import ModelPoolError, get_default_role_models


@dataclass(frozen=True)
class DebateConfig:
    base_url: str = "http://localhost:11434"
    api_mode: str = "ollama"  # "ollama", "openai", or "gemini"
    gemini_api_key: str | None = None  # Required if api_mode="gemini"
    moderator_model: str = "qwen3:30b"
    conspiracy_model: str = "gemma3:27b"
    scientific_model: str = "glm-4.7-flash:latest"
    moderator_temperature: float = 0.2
    conspiracy_temperature: float = 0.6
    scientific_temperature: float = 0.2
    max_tokens: int = 600
    thinking_budget: int = 4000  # Extra tokens reserved for thinking models' chain-of-thought
    num_ctx: int = 8192  # Ollama context window size (num_ctx)
    rounds: int = 3
    word_limit: int = 180
    seed: int | None = None
    max_search_rounds: int = 3
    output_dir: str = "results"
    run_analysis: bool = True
    analysis_shift_threshold: int = 5
    analysis_similarity_method: str = "tfidf"

    # Advanced Analysis (Research Mode)
    adv_analysis_emotion_model: str = "bhadresh-savani/bert-base-uncased-emotion"
    adv_analysis_overwrite: bool = True
    adv_analysis_max_runs: int = 10
    uncertainty_lexicon: list[str] = field(default_factory=lambda: sorted(list(UNCERTAINTY_WORDS)))
    strong_modality_lexicon: list[str] = field(default_factory=lambda: sorted(list(MODALITY_STRONG_WORDS)))
    weak_modality_lexicon: list[str] = field(default_factory=lambda: sorted(list(MODALITY_WEAK_WORDS)))

    @classmethod
    def from_ini(cls, config_path: str | Path = "config/config.ini") -> DebateConfig:
        """Load configuration from an INI file.

        Args:
            config_path: Path to the config.ini file.
                Defaults to ``config/config.ini`` relative to the working directory.

        Returns:
            DebateConfig instance with values from the INI file, falling back to defaults.
        """
        config = ConfigParser()
        config_path = Path(config_path)

        # Create a default instance to use as fallback for defaults
        default_cfg = cls()

        # Prefer model defaults from model_pool.json when available
        try:
            pooled_defaults = get_default_role_models()
        except ModelPoolError:
            pooled_defaults = {
                "moderator": default_cfg.moderator_model,
                "conspiracy": default_cfg.conspiracy_model,
                "scientific": default_cfg.scientific_model,
            }

        # If config file doesn't exist, return defaults
        if not config_path.exists():
            return default_cfg

        config.read(config_path)

        # Helper to safely get values with defaults
        def get_str(section: str, key: str, default: str) -> str:
            value = config.get(section, key, fallback=default)
            if not value or value.strip() == "":
                return default
            # ConfigParser does not strip quotes; remove wrapping '"' or "'".
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            return value

        def get_int(section: str, key: str, default: int) -> int:
            try:
                return config.getint(section, key, fallback=default)
            except ValueError:
                return default

        def get_float(section: str, key: str, default: float) -> float:
            try:
                return config.getfloat(section, key, fallback=default)
            except ValueError:
                return default

        def get_bool(section: str, key: str, default: bool) -> bool:
            try:
                return config.getboolean(section, key, fallback=default)
            except ValueError:
                return default

        def get_optional_int(section: str, key: str) -> int | None:
            value = config.get(section, key, fallback=None)
            if value is None or value.strip() == "":
                return None
            try:
                return int(value)
            except ValueError:
                return None

        def get_list(section: str, key: str, default: list[str]) -> list[str]:
            value = config.get(section, key, fallback=None)
            if value is None or value.strip() == "":
                return default
            return [item.strip() for item in value.split(",") if item.strip()]

        load_models_from_ini = get_bool("models", "load_from_ini", False)
        moderator_model = (
            get_str("models", "moderator_model", pooled_defaults["moderator"])
            if load_models_from_ini
            else pooled_defaults["moderator"]
        )
        conspiracy_model = (
            get_str("models", "conspiracy_model", pooled_defaults["conspiracy"])
            if load_models_from_ini
            else pooled_defaults["conspiracy"]
        )
        scientific_model = (
            get_str("models", "scientific_model", pooled_defaults["scientific"])
            if load_models_from_ini
            else pooled_defaults["scientific"]
        )

        return cls(
            base_url=get_str("api", "base_url", default_cfg.base_url),
            api_mode=get_str("api", "api_mode", default_cfg.api_mode),
            gemini_api_key=os.environ.get("GEMINI_API_KEY")
            or get_str("api", "gemini_api_key", "")
            or None,
            moderator_model=moderator_model,
            conspiracy_model=conspiracy_model,
            scientific_model=scientific_model,
            # moderator_temperature=get_float(
            #     "models", "moderator_temperature", default_cfg.moderator_temperature
            # ),
            # conspiracy_temperature=get_float(
            #     "models", "conspiracy_temperature", default_cfg.conspiracy_temperature
            # ),
            # scientific_temperature=get_float(
            #     "models", "scientific_temperature", default_cfg.scientific_temperature
            # ),
            max_tokens=get_int("debate", "max_tokens", default_cfg.max_tokens),
            thinking_budget=get_int("debate", "thinking_budget", default_cfg.thinking_budget),
            num_ctx=get_int("debate", "num_ctx", default_cfg.num_ctx),
            rounds=get_int("debate", "rounds", default_cfg.rounds),
            word_limit=get_int("debate", "word_limit", default_cfg.word_limit),
            seed=get_optional_int("debate", "seed"),
            max_search_rounds=get_int("debate", "max_search_rounds", default_cfg.max_search_rounds),
            output_dir=get_str("output", "output_dir", default_cfg.output_dir),
            run_analysis=get_bool("analysis", "run_analysis", default_cfg.run_analysis),
            analysis_shift_threshold=get_int(
                "analysis", "analysis_shift_threshold", default_cfg.analysis_shift_threshold
            ),
            analysis_similarity_method=get_str(
                "analysis", "analysis_similarity_method", default_cfg.analysis_similarity_method
            ),
            adv_analysis_emotion_model=get_str(
                "analysis", "emotion_model", default_cfg.adv_analysis_emotion_model
            ),
            adv_analysis_overwrite=get_bool(
                "analysis", "overwrite", default_cfg.adv_analysis_overwrite
            ),
            adv_analysis_max_runs=get_int(
                "analysis", "max_runs", default_cfg.adv_analysis_max_runs
            ),
            uncertainty_lexicon=get_list(
                "analysis", "uncertainty_lexicon", default_cfg.uncertainty_lexicon
            ),
            strong_modality_lexicon=get_list(
                "analysis", "strong_modality_lexicon", default_cfg.strong_modality_lexicon
            ),
            weak_modality_lexicon=get_list(
                "analysis", "weak_modality_lexicon", default_cfg.weak_modality_lexicon
            ),
        )

    @classmethod
    def from_env(cls) -> DebateConfig:
        """Legacy alias for :meth:`from_ini`.

        Does **not** read environment variables (except ``GEMINI_API_KEY``
        which ``from_ini`` handles). Kept for backward compatibility.
        """
        return cls.from_ini()
