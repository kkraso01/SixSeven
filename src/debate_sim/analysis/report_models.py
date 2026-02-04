from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class StanceShiftEvent(BaseModel):
    round: int
    agent: str
    delta: int
    confidence_before: int
    confidence_after: int


class StanceSummary(BaseModel):
    start_confidence: Dict[str, int]
    end_confidence: Dict[str, int]
    net_shift: Dict[str, int]
    per_round_confidence: Dict[str, List[int]]
    per_round_delta: Dict[str, List[int]]
    shift_events: List[StanceShiftEvent]


class TacticSummary(BaseModel):
    counts: Dict[str, Dict[str, int]]
    diversity: Dict[str, int]
    entropy: Dict[str, float]


class QualityAggregate(BaseModel):
    mean: float
    min: int
    max: int


class QualitySummary(BaseModel):
    per_round: Dict[str, List[int]]
    aggregates: Dict[str, QualityAggregate]
    trends: Dict[str, float]


class RedundancySummary(BaseModel):
    by_agent: Dict[str, float]
    overall: float
    method: str


class PersuasionMoment(BaseModel):
    round: int
    affected_agent: str
    delta: int
    excerpt: str
    why_flagged: str


class AnalysisReport(BaseModel):
    run_id: str
    topic: str
    motion: str
    rounds_completed: int
    stance_summary: StanceSummary
    tactic_summary: TacticSummary
    quality_summary: QualitySummary
    redundancy_summary: RedundancySummary
    persuasion_moments: List[PersuasionMoment]
    figures: List[str]
    safety_flags: List[str] = Field(default_factory=list)
    limitations: List[str]
    run_config: Optional[Dict[str, object]] = None


class RunCaseSummary(BaseModel):
    run_id: str
    ca_net_shift: int
    civility_mean: float
    tactic_diversity_ca: int
    model_signature: Optional[str] = None


class AggregateReport(BaseModel):
    runs_analyzed: int
    grouped_by: str
    distributions: Dict[str, List[float]]
    best_cases: List[RunCaseSummary]
    worst_cases: List[RunCaseSummary]
    figures: List[str]
    limitations: List[str]
    groups: Optional[Dict[str, List[str]]] = None
