"""Unit tests for the evaluation module."""

from __future__ import annotations

from debate.core.schemas import ModeratorRecap
from debate.simulator.engine.evaluation import build_metrics_table, stance_shift
from debate.simulator.engine.memory import initial_memory, update_agent_state


class TestStanceShift:
    def test_no_change(self) -> None:
        mem = initial_memory("t", "m")
        shift = stance_shift(mem, mem)
        assert shift == {"CA_delta": 0, "SA_delta": 0}

    def test_positive_shift(self) -> None:
        prev = initial_memory("t", "m")
        curr = update_agent_state(prev, "CA", 70, "evidence")
        shift = stance_shift(prev, curr)
        assert shift["CA_delta"] == 15  # 70 - 55


class TestBuildMetricsTable:
    def test_one_recap(self) -> None:
        recap = ModeratorRecap(
            round=1,
            summary_agreements=["a"],
            summary_disagreements=["d"],
            detected_fallacies_or_moves=["none"],
            civility_score=4,
            epistemic_quality_score=3,
            bridge_building_score=2,
            confidence_updates={"CA_delta": 0, "SA_delta": 0},
            next_round_questions=["q"],
        )
        table = build_metrics_table([recap])
        assert len(table) == 1
        assert table[0]["civility"] == 4
        assert table[0]["round"] == 1
