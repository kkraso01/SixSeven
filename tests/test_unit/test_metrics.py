"""Unit tests for analysis metrics functions."""

from __future__ import annotations

from debate.analysis.utils.metrics import (
    ROLE_TO_AGENT,
    quality_summary_from_scores,
    safety_flags,
    stance_summary_from_logs,
    tactic_summary_from_logs,
)
from debate.core.schemas import DebateLogItem, MemoryState
from debate.simulator.engine.memory import append_log, initial_memory


def _add_turn(
    mem: MemoryState, round_num: int, role: str, confidence: int, tactic: str = "evidence"
) -> MemoryState:
    """Helper to append a debate log entry with given parameters."""
    speaker_map = {"proponent": "pro", "opponent": "con", "moderator": "neutral"}
    item = DebateLogItem(
        debate_id="test",
        claim="motion",
        round=round_num,
        speaker_role=role,
        utterance=f"Utterance r{round_num} {role}",
        stance=speaker_map.get(role, "neutral"),
        confidence=confidence,
        tactic_used=tactic,
        tool_used="none",
    )
    return append_log(mem, item)


class TestRoleToAgent:
    def test_constant_has_expected_keys(self) -> None:
        assert ROLE_TO_AGENT == {"proponent": "CA", "opponent": "SA", "moderator": "MA"}


class TestStanceSummary:
    def test_empty_log(self) -> None:
        mem = initial_memory("t", "m")
        summary = stance_summary_from_logs(mem, shift_threshold=5)
        assert summary.net_shift == {"CA": 0, "SA": 0}
        assert summary.shift_events == []

    def test_single_round_no_shift(self) -> None:
        mem = initial_memory("t", "m")
        mem = _add_turn(mem, 1, "proponent", 60)
        mem = _add_turn(mem, 1, "opponent", 50)
        summary = stance_summary_from_logs(mem, shift_threshold=5)
        assert summary.start_confidence["CA"] == 60
        assert summary.end_confidence["CA"] == 60
        assert summary.net_shift["CA"] == 0

    def test_shift_event_detected(self) -> None:
        mem = initial_memory("t", "m")
        mem = _add_turn(mem, 1, "proponent", 60)
        mem = _add_turn(mem, 2, "proponent", 45)  # -15 shift
        summary = stance_summary_from_logs(mem, shift_threshold=10)
        assert len(summary.shift_events) == 1
        event = summary.shift_events[0]
        assert event.agent == "CA"
        assert event.delta == -15


class TestTacticSummary:
    def test_counts_tactics(self) -> None:
        mem = initial_memory("t", "m")
        mem = _add_turn(mem, 1, "proponent", 60, tactic="evidence")
        mem = _add_turn(mem, 1, "opponent", 50, tactic="appeal to authority")
        mem = _add_turn(mem, 2, "proponent", 55, tactic="evidence")
        summary = tactic_summary_from_logs(mem.debate_log)
        assert summary.counts["CA"]["evidence"] == 2
        assert summary.counts["SA"]["appeal to authority"] == 1
        assert summary.diversity["CA"] == 1
        assert summary.diversity["SA"] == 1


class TestQualitySummary:
    def test_quality_aggregates(self) -> None:
        summary = quality_summary_from_scores(
            civility=[3, 4, 5],
            epistemic=[2, 3, 4],
            bridge=[1, 2, 3],
        )
        assert summary.aggregates["civility"].mean == 4.0
        assert summary.aggregates["civility"].min == 3
        assert summary.aggregates["civility"].max == 5

    def test_empty_scores(self) -> None:
        summary = quality_summary_from_scores([], [], [])
        assert summary.aggregates["civility"].mean == 0.0


class TestSafetyFlags:
    def test_no_flags_for_clean_text(self) -> None:
        mem = initial_memory("t", "m")
        mem = _add_turn(mem, 1, "proponent", 60)
        flags = safety_flags(mem)
        assert flags == []

    def test_flag_for_profanity(self) -> None:
        mem = initial_memory("t", "m")
        item = DebateLogItem(
            debate_id="test",
            claim="m",
            round=1,
            speaker_role="proponent",
            utterance="That is complete nonsense and you are an idiot",
            stance="pro",
            confidence=60,
            tactic_used="insult",
            tool_used="none",
        )
        mem = append_log(mem, item)
        flags = safety_flags(mem)
        assert len(flags) >= 1
