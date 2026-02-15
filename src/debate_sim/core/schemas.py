from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator


def _rename_keys(data: dict, renames: dict[str, str]) -> dict:
    """Apply a {wrong_key: right_key} rename map to *data* in-place.

    Only renames when the wrong key is present and the right key is absent,
    preventing silent data loss.
    """
    for wrong, right in renames.items():
        if wrong in data and right not in data:
            data[right] = data.pop(wrong)
    return data


def _fix_search_field(data: dict) -> dict:
    """Coerce a plain-string ``search`` value into a SearchRequest dict."""
    if "search" in data and isinstance(data["search"], str):
        data["search"] = {"should_search": True, "search_query": data["search"]}
    return data


class SearchRequest(BaseModel):
    """Request for web search during debate."""

    should_search: bool = Field(description="Whether to perform a web search")
    search_query: str | None = Field(default=None, description="The search query to execute")
    search_rationale: str | None = Field(default=None, description="Why this search is needed")


class SearchPlan(BaseModel):
    """Phase-1 output: agent decides whether to search before arguing."""

    should_search: bool = Field(description="Whether to perform a web search this turn")
    search_query: str | None = Field(
        default=None, description="The search query to execute (if searching)"
    )
    search_rationale: str | None = Field(
        default=None, description="Why this search is needed (if searching)"
    )


class AgentTurn(BaseModel):
    """Debate turn - agent argues their position but can be persuaded over time."""

    speaker: Literal["CA", "SA"]
    round: int
    tactic_used: str
    claim: str
    reasons: list[str] = Field(min_length=2)
    question_to_opponent: str
    confidence: Annotated[int, Field(ge=0, le=100)] = Field(
        description="Current belief conviction (0-100). Adjust based on opponent's arguments - lower if persuaded, higher if reinforced."
    )
    what_changes_mind: str = Field(
        description="One concrete type of evidence or condition that would change your position"
    )
    tone: str | None = None
    token_estimate: int | None = None
    search: SearchRequest | None = Field(default=None, description="Optional search request")

    @model_validator(mode="before")
    @classmethod
    def _fix_common_llm_mistakes(cls, data):
        """Fix common LLM JSON key naming errors before validation."""
        if isinstance(data, dict):
            _rename_keys(
                data,
                {
                    "question to opponent": "question_to_opponent",
                    "what changes mind": "what_changes_mind",
                    "tactic used": "tactic_used",
                    "token estimate": "token_estimate",
                },
            )
            _fix_search_field(data)
            # Coerce reasons: string -> list, single-item with semicolons -> split
            if "reasons" in data:
                r = data["reasons"]
                if isinstance(r, str):
                    data["reasons"] = [s.strip() for s in r.split(";") if s.strip()]
                elif isinstance(r, list) and len(r) == 1 and isinstance(r[0], str) and ";" in r[0]:
                    data["reasons"] = [s.strip() for s in r[0].split(";") if s.strip()]
            # Strip stray keys LLMs add (e.g. "title")
            allowed = {f.alias or name for name, f in cls.model_fields.items()}
            allowed.update(cls.model_fields.keys())
            # Don't strip for subclasses — they have extra fields
            if cls is AgentTurn:
                data = {k: v for k, v in data.items() if k in allowed}
        return data


class ScientificTurn(AgentTurn):
    clarify: str
    evaluate_gaps: list[str]
    alternative_hypotheses: list[str]
    discriminating_tests: list[str]

    @model_validator(mode="before")
    @classmethod
    def _fix_scientific_keys(cls, data):
        """Fix common LLM key naming errors for ScientificTurn fields."""
        if isinstance(data, dict):
            _rename_keys(
                data,
                {
                    "evaluate gaps": "evaluate_gaps",
                    "alternative hypotheses": "alternative_hypotheses",
                    "discriminating tests": "discriminating_tests",
                    "question to opponent": "question_to_opponent",
                    "what changes mind": "what_changes_mind",
                    "tactic used": "tactic_used",
                    "token estimate": "token_estimate",
                },
            )
            _fix_search_field(data)
            # Strip stray keys (e.g. "title")
            allowed = set(cls.model_fields.keys())
            data = {k: v for k, v in data.items() if k in allowed}
        return data


class ModeratorRecap(BaseModel):
    """Moderator's analysis of each round - tracks persuasion dynamics."""

    round: int
    summary_agreements: list[str]
    summary_disagreements: list[str]
    detected_fallacies_or_moves: list[str]
    civility_score: Annotated[int, Field(ge=0, le=5)]
    epistemic_quality_score: Annotated[int, Field(ge=0, le=5)]
    bridge_building_score: Annotated[int, Field(ge=0, le=5)]
    confidence_updates: dict[str, int] = Field(
        description="Confidence deltas: {'CA_delta': int, 'SA_delta': int}"
    )
    next_round_questions: list[str]

    @model_validator(mode="before")
    @classmethod
    def _fix_recap_keys(cls, data):
        """Fix common LLM key-naming variants for moderator recap fields."""
        if isinstance(data, dict):
            _rename_keys(
                data,
                {
                    "summary agreements": "summary_agreements",
                    "summary disagreements": "summary_disagreements",
                    "detected fallacies or moves": "detected_fallacies_or_moves",
                    "detected fallacies": "detected_fallacies_or_moves",
                    "civility score": "civility_score",
                    "epistemic quality score": "epistemic_quality_score",
                    "epistemic score": "epistemic_quality_score",
                    "bridge building score": "bridge_building_score",
                    "bridge score": "bridge_building_score",
                    "confidence updates": "confidence_updates",
                    "next round questions": "next_round_questions",
                },
            )
            # Coerce string lists to actual lists
            for field in [
                "summary_agreements",
                "summary_disagreements",
                "detected_fallacies_or_moves",
                "next_round_questions",
            ]:
                if field in data and isinstance(data[field], str):
                    data[field] = [s.strip() for s in data[field].split(",") if s.strip()]
            # Ensure confidence_updates is a dict with expected keys
            cu = data.get("confidence_updates")
            if cu is None:
                data["confidence_updates"] = {"CA_delta": 0, "SA_delta": 0}
            elif isinstance(cu, (int, float)):
                data["confidence_updates"] = {"CA_delta": int(cu), "SA_delta": int(cu)}
        return data


class ModeratorDecision(BaseModel):
    """Moderator's decision on whether to continue or end the debate."""

    should_continue: bool = Field(
        description="Whether the debate should continue to the next round"
    )
    reason: str = Field(description="Explanation for the decision")
    detected_mind_change: str | None = Field(
        default=None,
        description="Which agent (CA or SA) has significantly changed their mind, if any",
    )
    confidence_threshold_met: bool = Field(
        default=False,
        description="Whether confidence changes indicate a significant shift in position",
    )

    @model_validator(mode="before")
    @classmethod
    def _fix_decision_keys(cls, data):
        """Fix common LLM key-naming variants for moderator decision fields."""
        if isinstance(data, dict):
            _rename_keys(
                data,
                {
                    "should continue": "should_continue",
                    "detected mind change": "detected_mind_change",
                    "confidence threshold met": "confidence_threshold_met",
                },
            )
        return data


class DebateLogItem(BaseModel):
    """Canonical debate log format matching project specifications."""

    debate_id: str = Field(description="Unique identifier for this debate session")
    claim: str = Field(description="The motion/claim being debated")
    round: int = Field(description="Round number")
    speaker_role: Literal["proponent", "opponent", "moderator"] = Field(
        description="Role of the speaker in the debate"
    )
    utterance: str = Field(description="The actual content/statement made")
    stance: Literal["pro", "con", "neutral"] = Field(description="Position taken in this utterance")
    confidence: int | None = Field(default=None, description="Confidence level (0-100)")
    tactic_used: str | None = Field(default=None, description="Rhetorical tactic employed")
    tool_used: Literal["none", "tavily", "duckduckgo"] = Field(
        default="none", description="External tool used for this turn"
    )
    tool_query: str | None = Field(default=None, description="Query sent to external tool")
    reply_to_turn: int | None = Field(
        default=None, description="Turn number this is replying to"
    )


class AgentState(BaseModel):
    """Agent's epistemic state - tracks belief conviction over time."""

    confidence: int = Field(
        description="Current belief conviction (0-100). Changes as agent is persuaded or reinforced."
    )
    values: list[str]
    preferred_tactics: list[str]
    rejected_frames: list[str]
    what_changes_mind: str = Field(description="What evidence would change this agent's position")


class Scoreboard(BaseModel):
    stance_shift: dict[str, int]
    bridge_score: int
    civility_score: int
    epistemic_quality: int


class MemoryState(BaseModel):
    topic: str
    motion: str
    round: int
    agent_states: dict[str, AgentState]
    debate_log: list[DebateLogItem]
    scoreboard: Scoreboard
    moderator_notes: list[str]
    next_round_strategy: dict[str, str]


class PersuasionMoment(BaseModel):
    round: int
    speaker: str
    excerpt: str
    why_it_mattered: str

    @model_validator(mode="before")
    @classmethod
    def _fix_moment_keys(cls, data):
        """Fix common LLM key-naming variants for persuasion-moment fields."""
        if isinstance(data, dict):
            _rename_keys(data, {"why it mattered": "why_it_mattered"})
        return data


class FinalReport(BaseModel):
    """Final debate report - tracks persuasion outcomes."""

    topic: str
    motion: str
    rounds_completed: int
    stance_trajectory: dict[str, list[int]] = Field(
        description="Confidence over time: {'CA': [80, 75, 60], 'SA': [55, 60, 70]}"
    )
    tactic_counts: dict[str, int]
    key_persuasion_moments: list[PersuasionMoment]
    outcome_summary: str = Field(description="Who changed their mind, or if neither did")
    limitations: list[str]

    @model_validator(mode="before")
    @classmethod
    def _fix_report_keys(cls, data):
        """Fix common LLM key-naming variants for final-report fields."""
        if isinstance(data, dict):
            _rename_keys(
                data,
                {
                    "rounds completed": "rounds_completed",
                    "stance trajectory": "stance_trajectory",
                    "tactic counts": "tactic_counts",
                    "key persuasion moments": "key_persuasion_moments",
                    "outcome summary": "outcome_summary",
                },
            )
            # Ensure lists default to empty rather than crashing
            if "key_persuasion_moments" not in data:
                data["key_persuasion_moments"] = []
            if "limitations" not in data:
                data["limitations"] = []
            if "tactic_counts" not in data:
                data["tactic_counts"] = {}
            if "stance_trajectory" not in data:
                data["stance_trajectory"] = {"CA": [], "SA": []}
        return data
