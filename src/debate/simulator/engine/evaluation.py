"""Debate evaluation helpers — stance-shift computation and per-round metrics."""

from __future__ import annotations

from debate.core.schemas import MemoryState, ModeratorRecap


def stance_shift(previous: MemoryState, current: MemoryState) -> dict[str, int]:
    """Return per-agent confidence deltas between two memory snapshots."""
    return {
        "CA_delta": current.agent_states["CA"].confidence - previous.agent_states["CA"].confidence,
        "SA_delta": current.agent_states["SA"].confidence - previous.agent_states["SA"].confidence,
    }


def metrics_row(round_number: int, recap: ModeratorRecap) -> dict[str, int]:
    """Build a single-row dict for the per-round metrics CSV."""
    return {
        "round": round_number,
        "civility": recap.civility_score,
        "epistemic_quality": recap.epistemic_quality_score,
        "bridge": recap.bridge_building_score,
    }


def build_metrics_table(recap_history: list[ModeratorRecap]) -> list[dict[str, int]]:
    """Convert a list of moderator recaps into a list of metrics-row dicts."""
    return [metrics_row(idx + 1, recap) for idx, recap in enumerate(recap_history)]
