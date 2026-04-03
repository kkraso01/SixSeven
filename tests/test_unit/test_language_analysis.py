from __future__ import annotations

from debate.analysis.language_analysis import language_use_summary_from_logs
from debate.core.schemas import DebateLogItem


def _item(role: str, text: str) -> DebateLogItem:
    stance = "neutral" if role == "moderator" else ("pro" if role == "proponent" else "con")
    return DebateLogItem(
        debate_id="d1",
        claim="c",
        round=1,
        speaker_role=role,
        utterance=text,
        stance=stance,
        confidence=50,
        tool_used="none",
    )


def test_language_use_summary_counts_basic_dimensions() -> None:
    logs = [
        _item("proponent", "Maybe this is definitely a danger and threat to safety."),
        _item("opponent", "Evidence may suggest trust and credible expert consensus."),
    ]

    summary = language_use_summary_from_logs(logs)

    assert summary.token_count["CA"] > 0
    assert summary.token_count["SA"] > 0
    assert summary.uncertainty.overall_count >= 2
    assert summary.strong_modality.overall_count >= 1
    assert summary.weak_modality.overall_count >= 1
    assert "care_harm" in summary.moral_framing
    assert "fear" in summary.emotion


def test_language_use_summary_ignores_moderator() -> None:
    logs = [
        _item("moderator", "Maybe everyone should calm down."),
    ]

    summary = language_use_summary_from_logs(logs)

    assert summary.token_count == {"CA": 0, "SA": 0}
    assert summary.uncertainty.overall_count == 0
