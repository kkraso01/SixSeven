from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, conint


class AgentTurn(BaseModel):
    speaker: Literal["CA", "SA"]
    round: int
    tactic_used: str
    claim: str
    reasons: List[str] = Field(min_length=2)
    question_to_opponent: str
    confidence: conint(ge=0, le=100)
    what_changes_mind: str
    tone: Optional[str] = None
    token_estimate: Optional[int] = None


class ScientificTurn(AgentTurn):
    clarify: str
    evaluate_gaps: List[str]
    alternative_hypotheses: List[str]
    discriminating_tests: List[str]


class ModeratorRecap(BaseModel):
    round: int
    summary_agreements: List[str]
    summary_disagreements: List[str]
    detected_fallacies_or_moves: List[str]
    civility_score: conint(ge=0, le=5)
    epistemic_quality_score: conint(ge=0, le=5)
    bridge_building_score: conint(ge=0, le=5)
    confidence_updates: Dict[str, int]
    next_round_questions: List[str]


class DebateLogItem(BaseModel):
    round: int
    speaker: str
    content: str
    tactic_used: str
    confidence: int


class AgentState(BaseModel):
    confidence: int
    values: List[str]
    preferred_tactics: List[str]
    rejected_frames: List[str]
    what_changes_mind: str


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


class FinalReport(BaseModel):
    topic: str
    motion: str
    rounds_completed: int
    stance_trajectory: Dict[str, List[int]]
    tactic_counts: Dict[str, int]
    key_persuasion_moments: List[PersuasionMoment]
    outcome_summary: str
    limitations: List[str]
