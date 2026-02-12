from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, conint, model_validator


class SearchRequest(BaseModel):
    """Request for web search during debate."""
    should_search: bool = Field(description="Whether to perform a web search")
    search_query: Optional[str] = Field(default=None, description="The search query to execute")
    search_rationale: Optional[str] = Field(default=None, description="Why this search is needed")


class AgentTurn(BaseModel):
    """Debate turn - agent argues their position but can be persuaded over time."""
    speaker: Literal["CA", "SA"]
    round: int
    tactic_used: str
    claim: str
    reasons: List[str] = Field(min_length=2)
    question_to_opponent: str
    confidence: conint(ge=0, le=100) = Field(description="Current belief conviction (0-100). Adjust based on opponent's arguments - lower if persuaded, higher if reinforced.")
    what_changes_mind: str = Field(description="One concrete type of evidence or condition that would change your position")
    tone: Optional[str] = None
    token_estimate: Optional[int] = None
    search: Optional[SearchRequest] = Field(default=None, description="Optional search request")

    @model_validator(mode="before")
    @classmethod
    def _fix_common_llm_mistakes(cls, data):
        """Fix common LLM JSON key naming errors before validation."""
        if isinstance(data, dict):
            # Fix "question to opponent" -> "question_to_opponent"
            if "question to opponent" in data and "question_to_opponent" not in data:
                data["question_to_opponent"] = data.pop("question to opponent")
            # Fix "what changes mind" -> "what_changes_mind"
            if "what changes mind" in data and "what_changes_mind" not in data:
                data["what_changes_mind"] = data.pop("what changes mind")
            # Fix "tactic used" -> "tactic_used"
            if "tactic used" in data and "tactic_used" not in data:
                data["tactic_used"] = data.pop("tactic used")
            # Fix "token estimate" -> "token_estimate"
            if "token estimate" in data and "token_estimate" not in data:
                data["token_estimate"] = data.pop("token estimate")
            # Fix search being a plain string instead of SearchRequest object
            if "search" in data and isinstance(data["search"], str):
                data["search"] = {"should_search": True, "search_query": data["search"]}
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
    evaluate_gaps: List[str]
    alternative_hypotheses: List[str]
    discriminating_tests: List[str]

    @model_validator(mode="before")
    @classmethod
    def _fix_scientific_keys(cls, data):
        """Fix common LLM key naming errors for ScientificTurn fields."""
        if isinstance(data, dict):
            # Fix space-separated keys -> underscore
            renames = {
                "evaluate gaps": "evaluate_gaps",
                "alternative hypotheses": "alternative_hypotheses",
                "discriminating tests": "discriminating_tests",
                "question to opponent": "question_to_opponent",
                "what changes mind": "what_changes_mind",
                "tactic used": "tactic_used",
                "token estimate": "token_estimate",
            }
            for wrong, right in renames.items():
                if wrong in data and right not in data:
                    data[right] = data.pop(wrong)
            # Fix search being a plain string
            if "search" in data and isinstance(data["search"], str):
                data["search"] = {"should_search": True, "search_query": data["search"]}
            # Strip stray keys (e.g. "title")
            allowed = set(cls.model_fields.keys())
            data = {k: v for k, v in data.items() if k in allowed}
        return data


class ModeratorRecap(BaseModel):
    """Moderator's analysis of each round - tracks persuasion dynamics."""
    round: int
    summary_agreements: List[str]
    summary_disagreements: List[str]
    detected_fallacies_or_moves: List[str]
    civility_score: conint(ge=0, le=5)
    epistemic_quality_score: conint(ge=0, le=5)
    bridge_building_score: conint(ge=0, le=5)
    confidence_updates: Dict[str, int] = Field(description="Confidence deltas: {'CA_delta': int, 'SA_delta': int}")
    next_round_questions: List[str]

    @model_validator(mode="before")
    @classmethod
    def _fix_recap_keys(cls, data):
        if isinstance(data, dict):
            renames = {
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
            }
            for wrong, right in renames.items():
                if wrong in data and right not in data:
                    data[right] = data.pop(wrong)
            # Coerce string lists to actual lists
            for field in ["summary_agreements", "summary_disagreements", "detected_fallacies_or_moves", "next_round_questions"]:
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
    should_continue: bool = Field(description="Whether the debate should continue to the next round")
    reason: str = Field(description="Explanation for the decision")
    detected_mind_change: Optional[str] = Field(
        default=None, 
        description="Which agent (CA or SA) has significantly changed their mind, if any"
    )
    confidence_threshold_met: bool = Field(
        default=False,
        description="Whether confidence changes indicate a significant shift in position"
    )

    @model_validator(mode="before")
    @classmethod
    def _fix_decision_keys(cls, data):
        if isinstance(data, dict):
            renames = {
                "should continue": "should_continue",
                "detected mind change": "detected_mind_change",
                "confidence threshold met": "confidence_threshold_met",
            }
            for wrong, right in renames.items():
                if wrong in data and right not in data:
                    data[right] = data.pop(wrong)
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
    stance: Literal["pro", "con", "neutral"] = Field(
        description="Position taken in this utterance"
    )
    confidence: Optional[int] = Field(default=None, description="Confidence level (0-100)")
    tactic_used: Optional[str] = Field(default=None, description="Rhetorical tactic employed")
    tool_used: Literal["none", "tavily", "duckduckgo"] = Field(
        default="none",
        description="External tool used for this turn"
    )
    tool_query: Optional[str] = Field(default=None, description="Query sent to external tool")
    reply_to_turn: Optional[int] = Field(
        default=None,
        description="Turn number this is replying to"
    )


class AgentState(BaseModel):
    """Agent's epistemic state - tracks belief conviction over time."""
    confidence: int = Field(description="Current belief conviction (0-100). Changes as agent is persuaded or reinforced.")
    values: List[str]
    preferred_tactics: List[str]
    rejected_frames: List[str]
    what_changes_mind: str = Field(description="What evidence would change this agent's position")


class Scoreboard(BaseModel):
    stance_shift: Dict[str, int]
    bridge_score: int
    civility_score: int
    epistemic_quality: int


class MemoryState(BaseModel):
    topic: str
    motion: str
    round: int
    agent_states: Dict[str, AgentState]
    debate_log: List[DebateLogItem]
    scoreboard: Scoreboard
    moderator_notes: List[str]
    next_round_strategy: Dict[str, str]


class PersuasionMoment(BaseModel):
    round: int
    speaker: str
    excerpt: str
    why_it_mattered: str

    @model_validator(mode="before")
    @classmethod
    def _fix_moment_keys(cls, data):
        if isinstance(data, dict):
            renames = {
                "why it mattered": "why_it_mattered",
            }
            for wrong, right in renames.items():
                if wrong in data and right not in data:
                    data[right] = data.pop(wrong)
        return data


class FinalReport(BaseModel):
    """Final debate report - tracks persuasion outcomes."""
    topic: str
    motion: str
    rounds_completed: int
    stance_trajectory: Dict[str, List[int]] = Field(description="Confidence over time: {'CA': [80, 75, 60], 'SA': [55, 60, 70]}")
    tactic_counts: Dict[str, int]
    key_persuasion_moments: List[PersuasionMoment]
    outcome_summary: str = Field(description="Who changed their mind, or if neither did")
    limitations: List[str]

    @model_validator(mode="before")
    @classmethod
    def _fix_report_keys(cls, data):
        if isinstance(data, dict):
            renames = {
                "rounds completed": "rounds_completed",
                "stance trajectory": "stance_trajectory",
                "tactic counts": "tactic_counts",
                "key persuasion moments": "key_persuasion_moments",
                "outcome summary": "outcome_summary",
            }
            for wrong, right in renames.items():
                if wrong in data and right not in data:
                    data[right] = data.pop(wrong)
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
