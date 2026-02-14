"""Debate simulator package."""

from .analysis.analysis_runner import analyze_all, analyze_run
from .core.config import DebateConfig
from .core.container import DebateServices, build_default_services
from .core.protocols import (
    ArtifactExporter,
    DebateAnalyzer,
    LLMClient,
    PromptLoader,
    SearchProvider,
    StructuredLLMService,
)
from .debate.orchestrator import run_debate

__all__ = [
    "DebateConfig",
    "DebateServices",
    "build_default_services",
    "run_debate",
    "analyze_run",
    "analyze_all",
    # Protocols (for custom implementations)
    "LLMClient",
    "StructuredLLMService",
    "SearchProvider",
    "PromptLoader",
    "ArtifactExporter",
    "DebateAnalyzer",
]
