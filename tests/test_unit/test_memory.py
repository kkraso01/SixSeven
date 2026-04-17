"""Unit tests for memory state management functions."""

from __future__ import annotations

from debate.core.schemas import DebateLogItem, MemoryState
from debate.simulator.engine.memory import (
    append_log,
    initial_memory,
    update_agent_state,
    update_round,
    update_scoreboard,
)


class TestInitialMemory:
    """initial_memory should set up a valid starting state."""

    def test_returns_memory_state(self) -> None:
        mem = initial_memory("topic", "motion")
        assert isinstance(mem, MemoryState)

    def test_initial_round_is_zero(self) -> None:
        mem = initial_memory("t", "m")
        assert mem.round == 0

    def test_both_agents_present(self) -> None:
        mem = initial_memory("t", "m")
        assert "CA" in mem.agent_states
        assert "SA" in mem.agent_states

    def test_initial_confidence(self) -> None:
        mem = initial_memory("t", "m")
        assert mem.agent_states["CA"].confidence == 55
        assert mem.agent_states["SA"].confidence == 55

    def test_debate_log_empty(self) -> None:
        mem = initial_memory("t", "m")
        assert mem.debate_log == []


class TestAppendLog:
    def test_append_preserves_immutability(self) -> None:
        mem = initial_memory("t", "m")
        item = _make_log_item(round_num=1, speaker_role="proponent")
        new_mem = append_log(mem, item)
        assert len(new_mem.debate_log) == 1
        assert len(mem.debate_log) == 0  # original unchanged


class TestUpdateAgentState:
    def test_confidence_updated(self) -> None:
        mem = initial_memory("t", "m")
        new_mem = update_agent_state(mem, "CA", 80, "new evidence")
        assert new_mem.agent_states["CA"].confidence == 80
        assert new_mem.agent_states["CA"].what_changes_mind == "new evidence"
        # SA unchanged
        assert new_mem.agent_states["SA"].confidence == 55


class TestUpdateRound:
    def test_round_incremented(self) -> None:
        mem = initial_memory("t", "m")
        new_mem = update_round(mem, 3)
        assert new_mem.round == 3
        assert mem.round == 0  # original unchanged


class TestUpdateScoreboard:
    def test_scoreboard_replaced(self) -> None:
        mem = initial_memory("t", "m")
        new_mem = update_scoreboard(
            mem,
            stance_shift={"CA_delta": 5, "SA_delta": -3},
            bridge_score=4,
            civility_score=3,
            epistemic_quality=5,
        )
        assert new_mem.scoreboard.bridge_score == 4
        assert new_mem.scoreboard.stance_shift["CA_delta"] == 5


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_log_item(
    round_num: int = 1,
    speaker_role: str = "proponent",
) -> DebateLogItem:
    return DebateLogItem(
        debate_id="test_001",
        claim="Test motion",
        round=round_num,
        speaker_role=speaker_role,
        utterance="Test utterance",
        stance="pro",
        confidence=60,
        tactic_used="evidence",
        tool_used="none",
    )
