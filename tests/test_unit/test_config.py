"""Unit tests for core configuration loading."""

from __future__ import annotations

import textwrap
from pathlib import Path

from debate_sim.core.config import DebateConfig


class TestDebateConfigDefaults:
    """DebateConfig should return sensible defaults when no file is found."""

    def test_defaults_when_no_file(self, tmp_path: Path) -> None:
        cfg = DebateConfig.from_ini(tmp_path / "nonexistent.ini")
        assert cfg.api_mode == "ollama"
        assert cfg.rounds == 3
        assert cfg.output_dir == "artifacts"

    def test_from_env_delegates_to_from_ini(self) -> None:
        # from_env() is a legacy alias — should not raise
        cfg = DebateConfig.from_env()
        assert isinstance(cfg, DebateConfig)


class TestDebateConfigIniParsing:
    """DebateConfig.from_ini should parse INI values correctly."""

    def _write_ini(self, path: Path, content: str) -> Path:
        ini = path / "test_config.ini"
        ini.write_text(textwrap.dedent(content))
        return ini

    def test_basic_values(self, tmp_path: Path) -> None:
        ini = self._write_ini(
            tmp_path,
            """\
            [api]
            api_mode = gemini

            [debate]
            rounds = 7
            max_tokens = 1200

            [output]
            output_dir = my-artifacts
        """,
        )
        cfg = DebateConfig.from_ini(ini)
        assert cfg.api_mode == "gemini"
        assert cfg.rounds == 7
        assert cfg.max_tokens == 1200
        assert cfg.output_dir == "my-artifacts"

    def test_quoted_output_dir_stripped(self, tmp_path: Path) -> None:
        """Quotes around INI values should be stripped (ConfigParser doesn't do this)."""
        ini = self._write_ini(
            tmp_path,
            """\
            [output]
            output_dir = "quoted-dir"
        """,
        )
        cfg = DebateConfig.from_ini(ini)
        assert cfg.output_dir == "quoted-dir"

    def test_invalid_int_falls_back_to_default(self, tmp_path: Path) -> None:
        ini = self._write_ini(
            tmp_path,
            """\
            [debate]
            rounds = not-a-number
        """,
        )
        cfg = DebateConfig.from_ini(ini)
        assert cfg.rounds == DebateConfig.rounds  # falls back to default

    def test_optional_seed_empty_string(self, tmp_path: Path) -> None:
        ini = self._write_ini(
            tmp_path,
            """\
            [debate]
            seed =
        """,
        )
        cfg = DebateConfig.from_ini(ini)
        assert cfg.seed is None
