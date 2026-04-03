"""Pydantic models for analysis reports — stance, quality, tactics, redundancy."""

from __future__ import annotations

from pydantic import BaseModel, Field


class StanceShiftEvent(BaseModel):
    """A single significant confidence shift within a debate round."""

    round: int
    agent: str
    delta: int
    confidence_before: int
    confidence_after: int


class StanceSummary(BaseModel):
    """Aggregated stance/confidence trajectory for all agents."""

    start_confidence: dict[str, int]
    end_confidence: dict[str, int]
    net_shift: dict[str, int]
    per_round_confidence: dict[str, list[int]]
    per_round_delta: dict[str, list[int]]
    shift_events: list[StanceShiftEvent]


class TacticSummary(BaseModel):
    """Per-agent tactic usage counts, diversity, and entropy."""

    counts: dict[str, dict[str, int]]
    diversity: dict[str, int]
    entropy: dict[str, float]


class QualityAggregate(BaseModel):
    """Mean / min / max for a single quality dimension."""

    mean: float
    min: int
    max: int


class QualitySummary(BaseModel):
    """Per-round and aggregated quality scores."""

    per_round: dict[str, list[int]]
    aggregates: dict[str, QualityAggregate]
    trends: dict[str, float]


class RedundancySummary(BaseModel):
    """Pairwise text-similarity summary measuring argument repetition."""

    by_agent: dict[str, float]
    overall: float
    method: str


class PersuasionFlag(BaseModel):
    """A flagged persuasion moment detected during analysis."""

    round: int
    affected_agent: str
    delta: int
    excerpt: str
    why_flagged: str


class LexiconDimensionSummary(BaseModel):
    """Counts and normalized rates for one lexicon dimension."""

    counts: dict[str, int]
    rate_per_1000: dict[str, float]
    overall_count: int
    overall_rate_per_1000: float


class LanguageUseSummary(BaseModel):
    """Language feature summary aligned with project NLP objectives."""

    token_count: dict[str, int]
    uncertainty: LexiconDimensionSummary
    strong_modality: LexiconDimensionSummary
    weak_modality: LexiconDimensionSummary
    moral_framing: dict[str, LexiconDimensionSummary]
    emotion: dict[str, LexiconDimensionSummary]


class AnalysisReport(BaseModel):
    """Complete single-run analysis report."""

    run_id: str
    topic: str
    motion: str
    rounds_completed: int
    stance_summary: StanceSummary
    tactic_summary: TacticSummary
    quality_summary: QualitySummary
    redundancy_summary: RedundancySummary
    persuasion_moments: list[PersuasionFlag]
    language_use: LanguageUseSummary
    figures: list[str]
    safety_flags: list[str] = Field(default_factory=list)
    limitations: list[str]
    run_config: dict[str, object] | None = None


class RunCaseSummary(BaseModel):
    """Compact per-run summary used in aggregate reports."""

    run_id: str
    ca_net_shift: int
    civility_mean: float
    tactic_diversity_ca: int
    model_signature: str | None = None


class AggregateReport(BaseModel):
    """Cross-run aggregate analysis report."""

    runs_analyzed: int
    grouped_by: str
    distributions: dict[str, list[float]]
    best_cases: list[RunCaseSummary]
    worst_cases: list[RunCaseSummary]
    figures: list[str]
    limitations: list[str]
    groups: dict[str, list[str]] | None = None
