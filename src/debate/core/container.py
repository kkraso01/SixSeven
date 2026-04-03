"""Dependency-injection container and concrete adapter implementations.

The ``DebateServices`` dataclass aggregates every injectable dependency the
orchestrator needs.  Factory helpers (``build_default_services``,
``build_services_from_config``) wire the default concrete implementations so
callers only need a ``DebateConfig``.

Concrete adapters
─────────────────
* ``DuckDuckGoSearchProvider``  – wraps ``search_tool`` functions
* ``FilePromptLoader``          – wraps ``protocol.load_prompt``
* ``FileArtifactExporter``      – wraps ``export.writer.write_artifacts``
* ``DefaultDebateAnalyzer``     – wraps ``analysis_runner.analyze_run``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import DebateConfig
from .protocols import (
    ArtifactExporter,
    DebateAnalyzer,
    LLMClient,
    PromptLoader,
    SearchProvider,
    StructuredLLMService,
)
from .schemas import DebateLogItem, FinalReport, MemoryState, ModeratorRecap

# ── Concrete adapters ────────────────────────────────────────────────────────


class DuckDuckGoSearchProvider:
    """``SearchProvider`` backed by DuckDuckGo (via ``ddgs``)."""

    def search(self, query: str, max_results: int = 5) -> Any:
        from debate.simulator.providers.search import search_web

        return search_web(query, max_results=max_results)

    def format_results(self, response: Any, max_chars: int = 1000) -> str:
        from debate.simulator.providers.search import format_search_results_for_prompt

        return format_search_results_for_prompt(response, max_chars=max_chars)


class FilePromptLoader:
    """``PromptLoader`` that reads Markdown templates from a directory."""

    def __init__(self, prompt_dir: Path | None = None) -> None:
        if prompt_dir is None:
            prompt_dir = Path(__file__).resolve().parents[1] / "simulator" / "prompts"
        self._dir = prompt_dir

    def load(self, name: str, values: dict[str, str]) -> str:
        template = (self._dir / name).read_text(encoding="utf-8")
        return template.format(**values)


class FileArtifactExporter:
    """``ArtifactExporter`` that writes to the local filesystem."""

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
        from debate.simulator.io.writer import write_artifacts

        return write_artifacts(
            output_dir=output_dir,
            memory=memory,
            final_report=final_report,
            recaps=recaps,
            ca_sa_log=debate_log,
            metrics_table=metrics_table,
            topic=topic,
            motion=motion,
            run_config=run_config,
        )


class DefaultDebateAnalyzer:
    """``DebateAnalyzer`` using the built-in analysis runner."""

    def analyze(self, run_dir: str) -> Any:
        from debate.analysis.analysis_runner import analyze_run

        return analyze_run(run_dir)


# ── Service container ────────────────────────────────────────────────────────


@dataclass
class DebateServices:
    """Aggregates every dependency the orchestrator requires.

    All fields are typed against *protocols* so they can be replaced in
    tests or by different back-ends without touching the orchestrator.
    """

    llm: StructuredLLMService
    search: SearchProvider
    prompts: PromptLoader
    exporter: ArtifactExporter
    analyzer: DebateAnalyzer | None = None


# ── Factory helpers ──────────────────────────────────────────────────────────


def build_default_services(config: DebateConfig) -> DebateServices:
    """Wire the standard concrete implementations from a ``DebateConfig``.

    This is the single place that knows about concrete classes.
    """
    from debate.simulator.providers.instructor import StructuredLLM
    from debate.simulator.providers.llm_client import OllamaClient

    llm_client: LLMClient = OllamaClient(config)
    structured_llm: StructuredLLMService = StructuredLLM(llm_client)

    search: SearchProvider = DuckDuckGoSearchProvider()
    prompts: PromptLoader = FilePromptLoader()
    exporter: ArtifactExporter = FileArtifactExporter()
    analyzer: DebateAnalyzer | None = DefaultDebateAnalyzer() if config.run_analysis else None

    return DebateServices(
        llm=structured_llm,
        search=search,
        prompts=prompts,
        exporter=exporter,
        analyzer=analyzer,
    )
