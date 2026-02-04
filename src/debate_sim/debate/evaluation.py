from __future__ import annotations

from collections import Counter
from typing import Dict, List

from debate_sim.schemas import MemoryState, ModeratorRecap


def update_metrics(
    tactic_counts: Counter[str],
    recap: ModeratorRecap,
) -> Counter[str]:
    tactic_counts.update(recap.detected_fallacies_or_moves)
    return tactic_counts


def stance_shift(previous: MemoryState, current: MemoryState) -> Dict[str, int]:
    return {
        "CA_delta": current.agent_states["CA"].confidence
        - previous.agent_states["CA"].confidence,
        "SA_delta": current.agent_states["SA"].confidence
        - previous.agent_states["SA"].confidence,
    }


def metrics_row(round_number: int, recap: ModeratorRecap) -> Dict[str, int]:
    return {
        "round": round_number,
        "civility": recap.civility_score,
        "epistemic_quality": recap.epistemic_quality_score,
        "bridge": recap.bridge_building_score,
    }


def build_metrics_table(recap_history: List[ModeratorRecap]) -> List[Dict[str, int]]:
    return [metrics_row(idx + 1, recap) for idx, recap in enumerate(recap_history)]
