"""Protocol definitions for dependency injection.

All major subsystems are defined as Protocol classes so concrete
implementations can be swapped without touching the orchestrator.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

try:
    from typing import Protocol, runtime_checkable
except ImportError:  # Python < 3.8
    from typing_extensions import Protocol, runtime_checkable

from .schemas import DebateLogItem, FinalReport, MemoryState, ModeratorRecap  # noqa: E402

# ── LLM layer ───────────────────────────────────────────────────────────────


@runtime_checkable
class LLMClient(Protocol):
    """Low-level LLM client that produces structured Pydantic objects.

    Implementations: OllamaClient (Ollama / OpenAI-compat / Gemini).
    """

    def generate(
        self,
        response_model: type[T],
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        seed: int | None,
        validation_hint: str,
    ) -> T: ...


@runtime_checkable
class StructuredLLMService(Protocol):
    """High-level structured LLM service with retry / error-recovery logic.

    Implementations: StructuredLLM.
    """

    def call(
        self,
        response_model: type[T],
        messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        seed: int | None,
        max_retries: int = 5,
    ) -> T: ...


# ── Search layer ─────────────────────────────────────────────────────────────


@runtime_checkable
class SearchProvider(Protocol):
    """Web-search provider abstraction.

    Implementations: DuckDuckGoSearchProvider.
    """

    def search(self, query: str, max_results: int = 5) -> Any:
        """Return a SearchResponse-like object with .success / .results."""
        ...

    def format_results(self, response: Any, max_chars: int = 1000) -> str:
        """Format a SearchResponse into a prompt-friendly string."""
        ...


# ── Prompt layer ─────────────────────────────────────────────────────────────


@runtime_checkable
class PromptLoader(Protocol):
    """Loads and interpolates prompt templates.

    Implementations: FilePromptLoader (reads .md from disk).
    """

    def load(self, name: str, values: dict[str, str]) -> str: ...


# ── Export / Artifact layer ──────────────────────────────────────────────────


@runtime_checkable
class ArtifactExporter(Protocol):
    """Writes debate artefacts (transcripts, memory, reports, CSVs).

    Implementations: FileArtifactExporter.
    """

    def write(
        self,
        output_dir: Path,
        memory: MemoryState,
        final_report: FinalReport,
        recaps: list[ModeratorRecap],
        debate_log: list[DebateLogItem],
        metrics_table: list[dict],
        topic: str,
        motion: str,
        run_config: dict | None = None,
    ) -> Any:
        """Return an ExportBundle-like result."""
        ...


# ── Analysis layer ───────────────────────────────────────────────────────────


@runtime_checkable
class DebateAnalyzer(Protocol):
    """Runs post-debate analysis on a completed run directory.

    Implementations: DefaultDebateAnalyzer.
    """

    def analyze(self, run_dir: str) -> Any: ...
