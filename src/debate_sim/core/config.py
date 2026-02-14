from __future__ import annotations

import os
from configparser import ConfigParser
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DebateConfig:
    base_url: str = "http://localhost:11434"
    api_mode: str = "ollama"  # "ollama", "openai", or "gemini"
    gemini_api_key: str | None = None  # Required if api_mode="gemini"
    moderator_model: str = "llama3.1:8b"
    conspiracy_model: str = "llama3.1:8b"
    scientific_model: str = "llama3.1:8b"
    moderator_temperature: float = 0.2
    conspiracy_temperature: float = 0.6
    scientific_temperature: float = 0.2
    max_tokens: int = 600
    rounds: int = 3
    word_limit: int = 180
    seed: int | None = None
    output_dir: str = "artifacts"
    run_analysis: bool = True
    analysis_shift_threshold: int = 5
    analysis_similarity_method: str = "tfidf"

    # History management configuration
    history_mode: str = "global_full"  # "global_full", "per_agent", or "memory_only"
    include_memory_summary: bool = True  # Include compact memory summary
    history_trim: str = "none"  # "none", "rounds", "messages", or "chars"
    max_rounds_in_history: int | None = None  # Trim to last N rounds
    max_messages_in_history: int | None = None  # Trim to last N messages
    max_chars_in_history: int | None = None  # Trim to last N characters
    summarize_if_trimmed: bool = True  # Add moderator summary if history is trimmed
    highlight_opponent_last: bool = True  # Highlight opponent's last message

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

        # If config file doesn't exist, return defaults
        if not config_path.exists():
            return cls()

        config.read(config_path)

        # Helper to safely get values with defaults
        def get_str(section: str, key: str, default: str) -> str:
            value = config.get(section, key, fallback=default)
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

        return cls(
            base_url=get_str("api", "base_url", cls.base_url),
            api_mode=get_str("api", "api_mode", cls.api_mode),
            gemini_api_key=os.environ.get("GEMINI_API_KEY")
            or get_str("api", "gemini_api_key", "")
            or None,
            moderator_model=get_str("models", "moderator_model", cls.moderator_model),
            conspiracy_model=get_str("models", "conspiracy_model", cls.conspiracy_model),
            scientific_model=get_str("models", "scientific_model", cls.scientific_model),
            moderator_temperature=get_float(
                "models", "moderator_temperature", cls.moderator_temperature
            ),
            conspiracy_temperature=get_float(
                "models", "conspiracy_temperature", cls.conspiracy_temperature
            ),
            scientific_temperature=get_float(
                "models", "scientific_temperature", cls.scientific_temperature
            ),
            max_tokens=get_int("debate", "max_tokens", cls.max_tokens),
            rounds=get_int("debate", "rounds", cls.rounds),
            word_limit=get_int("debate", "word_limit", cls.word_limit),
            seed=get_optional_int("debate", "seed"),
            output_dir=get_str("output", "output_dir", cls.output_dir),
            run_analysis=get_bool("analysis", "run_analysis", cls.run_analysis),
            analysis_shift_threshold=get_int(
                "analysis", "analysis_shift_threshold", cls.analysis_shift_threshold
            ),
            analysis_similarity_method=get_str(
                "analysis", "analysis_similarity_method", cls.analysis_similarity_method
            ),
            history_mode=get_str("history", "history_mode", cls.history_mode),
            include_memory_summary=get_bool(
                "history", "include_memory_summary", cls.include_memory_summary
            ),
            history_trim=get_str("history", "history_trim", cls.history_trim),
            max_rounds_in_history=get_optional_int("history", "max_rounds_in_history"),
            max_messages_in_history=get_optional_int("history", "max_messages_in_history"),
            max_chars_in_history=get_optional_int("history", "max_chars_in_history"),
            summarize_if_trimmed=get_bool(
                "history", "summarize_if_trimmed", cls.summarize_if_trimmed
            ),
            highlight_opponent_last=get_bool(
                "history", "highlight_opponent_last", cls.highlight_opponent_last
            ),
        )

    @classmethod
    def from_env(cls) -> DebateConfig:
        """Legacy alias for :meth:`from_ini`.

        Does **not** read environment variables (except ``GEMINI_API_KEY``
        which ``from_ini`` handles). Kept for backward compatibility.
        """
        return cls.from_ini()
