"""Debate simulator package."""

from .analysis.analysis_runner import analyze_all, analyze_run
from .config import DebateConfig
from .debate.orchestrator import run_debate

__all__ = ["DebateConfig", "run_debate", "analyze_run", "analyze_all"]
