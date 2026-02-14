"""Debate analysis metrics — stance tracking, tactics, quality, redundancy, and safety."""

from __future__ import annotations

import importlib.util
import math
from collections.abc import Iterable

from ..core.schemas import DebateLogItem, MemoryState
from .features import (
    TranscriptData,
    detect_recap_keyword,
    excerpt_for_round,
    extract_recap_text,
    safety_flags_from_text,
)
from .report_models import (
    PersuasionFlag,
    QualityAggregate,
    QualitySummary,
    RedundancySummary,
    StanceShiftEvent,
    StanceSummary,
    TacticSummary,
)

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None

# Canonical mapping from debate-log speaker roles to agent codes.
# Defined once here to avoid duplication across metric functions.
ROLE_TO_AGENT: dict[str, str] = {"proponent": "CA", "opponent": "SA", "moderator": "MA"}


def stance_summary_from_logs(
    memory: MemoryState,
    shift_threshold: int,
) -> StanceSummary:
    """Compute per-agent confidence trajectories and significant shift events."""
    role_to_agent = ROLE_TO_AGENT

    per_round_confidence: dict[str, list[int]] = {"CA": [], "SA": []}
    for item in sorted(memory.debate_log, key=lambda x: (x.round, x.speaker_role)):
        agent = role_to_agent.get(item.speaker_role)
        if agent in per_round_confidence and item.confidence is not None:
            per_round_confidence[agent].append(item.confidence)

    per_round_delta: dict[str, list[int]] = {"CA": [], "SA": []}
    shift_events: list[StanceShiftEvent] = []
    start_confidence: dict[str, int] = {}
    end_confidence: dict[str, int] = {}
    net_shift: dict[str, int] = {}

    for agent, values in per_round_confidence.items():
        if not values:
            start_confidence[agent] = 0
            end_confidence[agent] = 0
            net_shift[agent] = 0
            continue
        start_confidence[agent] = values[0]
        end_confidence[agent] = values[-1]
        net_shift[agent] = values[-1] - values[0]
        deltas = []
        for idx in range(1, len(values)):
            delta = values[idx] - values[idx - 1]
            deltas.append(delta)
            if abs(delta) >= shift_threshold:
                shift_events.append(
                    StanceShiftEvent(
                        round=idx + 1,
                        agent=agent,
                        delta=delta,
                        confidence_before=values[idx - 1],
                        confidence_after=values[idx],
                    )
                )
        per_round_delta[agent] = deltas

    return StanceSummary(
        start_confidence=start_confidence,
        end_confidence=end_confidence,
        net_shift=net_shift,
        per_round_confidence=per_round_confidence,
        per_round_delta=per_round_delta,
        shift_events=shift_events,
    )


def tactic_summary_from_logs(logs: Iterable[DebateLogItem]) -> TacticSummary:
    """Count tactic usage per agent and compute diversity / entropy."""
    role_to_agent = ROLE_TO_AGENT

    counts: dict[str, dict[str, int]] = {"CA": {}, "SA": {}}
    for item in logs:
        agent = role_to_agent.get(item.speaker_role)
        if agent in counts and item.tactic_used:
            speaker_counts = counts[agent]
            speaker_counts[item.tactic_used] = speaker_counts.get(item.tactic_used, 0) + 1

    diversity = {agent: len(values) for agent, values in counts.items()}
    entropy = {agent: _entropy(values) for agent, values in counts.items()}

    return TacticSummary(counts=counts, diversity=diversity, entropy=entropy)


def _entropy(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for value in counts.values():
        probability = value / total
        entropy -= probability * math.log2(probability)
    return entropy


def quality_summary_from_scores(
    civility: list[int],
    epistemic: list[int],
    bridge: list[int],
) -> QualitySummary:
    per_round = {
        "civility": civility,
        "epistemic_quality": epistemic,
        "bridge_building": bridge,
    }
    aggregates = {
        "civility": _aggregate_scores(civility),
        "epistemic_quality": _aggregate_scores(epistemic),
        "bridge_building": _aggregate_scores(bridge),
    }
    trends = {
        "civility": _trend(civility),
        "epistemic_quality": _trend(epistemic),
        "bridge_building": _trend(bridge),
    }
    return QualitySummary(per_round=per_round, aggregates=aggregates, trends=trends)


def _aggregate_scores(values: list[int]) -> QualityAggregate:
    if not values:
        return QualityAggregate(mean=0.0, min=0, max=0)
    mean = sum(values) / len(values)
    return QualityAggregate(mean=mean, min=min(values), max=max(values))


def _trend(values: list[int]) -> float:
    if len(values) < 2:
        return 0.0
    return (values[-1] - values[0]) / (len(values) - 1)


def persuasion_moments(
    memory: MemoryState,
    transcript: TranscriptData,
    shift_threshold: int,
) -> list[PersuasionFlag]:
    """Identify rounds where confidence shifted beyond *shift_threshold* alongside moderator signals."""
    role_to_agent = ROLE_TO_AGENT

    moments: list[PersuasionFlag] = []
    per_round_confidence: dict[str, list[int]] = {"CA": [], "SA": []}
    per_round_content: dict[tuple[int, str], str] = {}

    for item in sorted(memory.debate_log, key=lambda x: (x.round, x.speaker_role)):
        agent = role_to_agent.get(item.speaker_role)
        if agent and item.confidence is not None:
            per_round_confidence[agent].append(item.confidence)
            per_round_content[(item.round, agent)] = item.utterance

    rounds = max((item.round for item in memory.debate_log), default=0)
    for round_number in range(2, rounds + 1):
        recap_text = extract_recap_text(transcript, round_number)
        recap_flag = detect_recap_keyword(recap_text)
        for agent in ["CA", "SA"]:
            values = per_round_confidence.get(agent, [])
            if len(values) < round_number:
                continue
            delta = values[round_number - 1] - values[round_number - 2]
            if abs(delta) >= shift_threshold and (recap_flag or not recap_text):
                fallback = per_round_content.get((round_number, agent), "")
                excerpt = excerpt_for_round(transcript, round_number, agent, fallback)
                why_flagged = (
                    "confidence change with moderator signal" if recap_flag else "confidence change"
                )
                moments.append(
                    PersuasionFlag(
                        round=round_number,
                        affected_agent=agent,
                        delta=delta,
                        excerpt=excerpt,
                        why_flagged=why_flagged,
                    )
                )
    return moments


def redundancy_summary(
    memory: MemoryState,
    similarity_method: str,
) -> RedundancySummary:
    """Compute average pairwise text similarity per agent to detect repetition."""
    role_to_agent = ROLE_TO_AGENT

    texts_by_agent: dict[str, list[str]] = {"CA": [], "SA": []}
    for item in sorted(memory.debate_log, key=lambda x: (x.round, x.speaker_role)):
        agent = role_to_agent.get(item.speaker_role)
        if agent in texts_by_agent:
            texts_by_agent[agent].append(item.utterance)

    by_agent = {
        agent: _average_similarity(texts, similarity_method)
        for agent, texts in texts_by_agent.items()
    }
    overall_values = [value for value in by_agent.values() if value is not None]
    overall = sum(overall_values) / len(overall_values) if overall_values else 0.0
    return RedundancySummary(by_agent=by_agent, overall=overall, method=similarity_method)


def _average_similarity(texts: list[str], similarity_method: str) -> float:
    if len(texts) < 2:
        return 0.0
    similarities = []
    for idx in range(1, len(texts)):
        similarities.append(_similarity(texts[idx - 1], texts[idx], similarity_method))
    return sum(similarities) / len(similarities) if similarities else 0.0


def _similarity(text_a: str, text_b: str, similarity_method: str) -> float:
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a and not tokens_b:
        return 0.0
    if similarity_method == "tfidf":
        return _tfidf_similarity(text_a, text_b)
    intersection = tokens_a.intersection(tokens_b)
    union = tokens_a.union(tokens_b)
    return len(intersection) / len(union) if union else 0.0


def _tokenize(text: str) -> set[str]:
    return {token.lower() for token in text.split() if token.strip()}


def _tfidf_similarity(text_a: str, text_b: str) -> float:
    if not HAS_SKLEARN:
        return _similarity(text_a, text_b, "jaccard")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform([text_a, text_b])
    similarity = cosine_similarity(matrix[0:1], matrix[1:2])
    return float(similarity[0][0])


def safety_flags(memory: MemoryState) -> list[str]:
    """Return a list of safety-flag strings for utterances containing profanity."""
    role_to_agent = ROLE_TO_AGENT

    log_texts = [
        (item.round, role_to_agent.get(item.speaker_role, item.speaker_role), item.utterance)
        for item in memory.debate_log
    ]
    return safety_flags_from_text(log_texts)
