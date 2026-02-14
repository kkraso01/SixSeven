"""Unit tests for Pydantic schema validators (the _fix_*_keys logic)."""

from __future__ import annotations

from debate_sim.core.schemas import (
    AgentTurn,
    FinalReport,
    ModeratorDecision,
    ModeratorRecap,
    PersuasionMoment,
    _rename_keys,
)


class TestRenameKeys:
    """Tests for the shared _rename_keys helper."""

    def test_renames_when_target_absent(self) -> None:
        data = {"wrong key": 42}
        _rename_keys(data, {"wrong key": "right_key"})
        assert data == {"right_key": 42}

    def test_no_rename_when_target_present(self) -> None:
        data = {"wrong key": 1, "right_key": 2}
        _rename_keys(data, {"wrong key": "right_key"})
        # Both keys remain — we don't clobber the existing correct key
        assert data == {"wrong key": 1, "right_key": 2}

    def test_noop_when_wrong_key_absent(self) -> None:
        data = {"other": 99}
        _rename_keys(data, {"missing": "also_missing"})
        assert data == {"other": 99}


class TestAgentTurnValidator:
    """AgentTurn._fix_common_llm_mistakes should normalise LLM output."""

    def _base_data(self, **overrides) -> dict:
        base = {
            "speaker": "CA",
            "round": 1,
            "tactic_used": "appeal to data",
            "claim": "Test claim",
            "reasons": ["r1", "r2"],
            "question_to_opponent": "Why?",
            "confidence": 60,
            "what_changes_mind": "evidence",
        }
        base.update(overrides)
        return base

    def test_space_separated_keys_fixed(self) -> None:
        data = self._base_data()
        data["question to opponent"] = data.pop("question_to_opponent")
        data["what changes mind"] = data.pop("what_changes_mind")
        turn = AgentTurn.model_validate(data)
        assert turn.question_to_opponent == "Why?"
        assert turn.what_changes_mind == "evidence"

    def test_reasons_semicolon_string_split(self) -> None:
        data = self._base_data(reasons="first; second; third")
        turn = AgentTurn.model_validate(data)
        assert turn.reasons == ["first", "second", "third"]

    def test_reasons_single_item_semicolon_split(self) -> None:
        data = self._base_data(reasons=["foo; bar"])
        turn = AgentTurn.model_validate(data)
        assert turn.reasons == ["foo", "bar"]

    def test_search_plain_string_coerced(self) -> None:
        data = self._base_data(search="vaccine efficacy")
        turn = AgentTurn.model_validate(data)
        assert turn.search is not None
        assert turn.search.should_search is True
        assert turn.search.search_query == "vaccine efficacy"

    def test_stray_keys_stripped(self) -> None:
        data = self._base_data(title="DO NOT INCLUDE")
        turn = AgentTurn.model_validate(data)
        # 'title' is not on the schema — should be silently dropped
        assert not hasattr(turn, "title")


class TestModeratorRecapValidator:
    def _base_recap(self) -> dict:
        return {
            "round": 1,
            "summary_agreements": ["a"],
            "summary_disagreements": ["d"],
            "detected_fallacies_or_moves": ["none"],
            "civility_score": 4,
            "epistemic_quality_score": 3,
            "bridge_building_score": 2,
            "confidence_updates": {"CA_delta": 0, "SA_delta": 0},
            "next_round_questions": ["q"],
        }

    def test_space_keys_fixed(self) -> None:
        data = self._base_recap()
        data["summary agreements"] = data.pop("summary_agreements")
        data["civility score"] = data.pop("civility_score")
        recap = ModeratorRecap.model_validate(data)
        assert recap.civility_score == 4

    def test_confidence_updates_defaults_when_none(self) -> None:
        data = self._base_recap()
        data["confidence_updates"] = None
        recap = ModeratorRecap.model_validate(data)
        assert recap.confidence_updates == {"CA_delta": 0, "SA_delta": 0}


class TestModeratorDecisionValidator:
    def test_space_keys(self) -> None:
        data = {
            "should continue": False,
            "reason": "done",
        }
        decision = ModeratorDecision.model_validate(data)
        assert decision.should_continue is False


class TestFinalReportValidator:
    def test_defaults_for_missing_lists(self) -> None:
        data = {
            "topic": "t",
            "motion": "m",
            "rounds_completed": 3,
            "outcome_summary": "tie",
        }
        report = FinalReport.model_validate(data)
        assert report.key_persuasion_moments == []
        assert report.limitations == []
        assert report.tactic_counts == {}
        assert report.stance_trajectory == {"CA": [], "SA": []}


class TestPersuasionMomentValidator:
    def test_space_key_fixed(self) -> None:
        data = {
            "round": 2,
            "speaker": "CA",
            "excerpt": "...",
            "why it mattered": "compelling evidence",
        }
        moment = PersuasionMoment.model_validate(data)
        assert moment.why_it_mattered == "compelling evidence"
