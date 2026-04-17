"""Lexicon-based language analysis for debate logs."""

from __future__ import annotations

import re
from collections import Counter

from debate.core.schemas import DebateLogItem

from .lexicons import (
    EMOTION_LEXICON,
    MODALITY_STRONG_WORDS,
    MODALITY_WEAK_WORDS,
    MORAL_FOUNDATION_LEXICON,
    UNCERTAINTY_WORDS,
)
from .metrics import ROLE_TO_AGENT
from .report_models import LanguageUseSummary, LexiconDimensionSummary

TOKEN_PATTERN: re.Pattern[str] = re.compile(r"[a-zA-Z']+")


def language_use_summary_from_logs(
    logs: list[DebateLogItem],
    uncertainty_lexicon: list[str] | set[str] | None = None,
    strong_modality_lexicon: list[str] | set[str] | None = None,
    weak_modality_lexicon: list[str] | set[str] | None = None,
) -> LanguageUseSummary:
    """Compute uncertainty, modality, moral framing, and emotion summaries."""
    token_counts: dict[str, int] = {"CA": 0, "SA": 0}
    tokens_by_agent: dict[str, list[str]] = {"CA": [], "SA": []}

    for item in logs:
        agent = ROLE_TO_AGENT.get(item.speaker_role)
        if agent not in tokens_by_agent:
            continue
        tokens = _tokenize(item.utterance)
        tokens_by_agent[agent].extend(tokens)
        token_counts[agent] += len(tokens)

    # Use defaults if not provided
    uncertainty_set = (
        set(uncertainty_lexicon) if uncertainty_lexicon is not None else UNCERTAINTY_WORDS
    )
    strong_set = (
        set(strong_modality_lexicon)
        if strong_modality_lexicon is not None
        else MODALITY_STRONG_WORDS
    )
    weak_set = (
        set(weak_modality_lexicon) if weak_modality_lexicon is not None else MODALITY_WEAK_WORDS
    )

    uncertainty = _summarize_dimension(tokens_by_agent, token_counts, uncertainty_set)
    strong_modality = _summarize_dimension(tokens_by_agent, token_counts, strong_set)
    weak_modality = _summarize_dimension(tokens_by_agent, token_counts, weak_set)
    moral = _summarize_categories(tokens_by_agent, token_counts, MORAL_FOUNDATION_LEXICON)
    emotion = _summarize_categories(tokens_by_agent, token_counts, EMOTION_LEXICON)

    return LanguageUseSummary(
        token_count=token_counts,
        uncertainty=uncertainty,
        strong_modality=strong_modality,
        weak_modality=weak_modality,
        moral_framing=moral,
        emotion=emotion,
    )


def _tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def _summarize_dimension(
    tokens_by_agent: dict[str, list[str]], token_counts: dict[str, int], lexicon: set[str]
) -> LexiconDimensionSummary:
    counts = {
        agent: sum(1 for token in tokens if token in lexicon)
        for agent, tokens in tokens_by_agent.items()
    }
    rates = {
        agent: _rate_per_1000(counts[agent], token_counts[agent])
        for agent in tokens_by_agent.keys()
    }
    total_count = sum(counts.values())
    total_tokens = sum(token_counts.values())
    return LexiconDimensionSummary(
        counts=counts,
        rate_per_1000=rates,
        overall_count=total_count,
        overall_rate_per_1000=_rate_per_1000(total_count, total_tokens),
    )


def _summarize_categories(
    tokens_by_agent: dict[str, list[str]],
    token_counts: dict[str, int],
    category_lexicon: dict[str, set[str]],
) -> dict[str, LexiconDimensionSummary]:
    result: dict[str, LexiconDimensionSummary] = {}
    for category, lexicon in category_lexicon.items():
        result[category] = _summarize_dimension(tokens_by_agent, token_counts, lexicon)
    return result


def _rate_per_1000(count: int, total_tokens: int) -> float:
    if total_tokens <= 0:
        return 0.0
    return (count / total_tokens) * 1000.0


def top_terms_by_agent(logs: list[DebateLogItem], top_k: int = 10) -> dict[str, list[tuple[str, int]]]:
    """Return top non-trivial terms per debating agent for optional diagnostics."""
    terms: dict[str, Counter[str]] = {"CA": Counter(), "SA": Counter()}
    for item in logs:
        agent = ROLE_TO_AGENT.get(item.speaker_role)
        if agent not in terms:
            continue
        for token in _tokenize(item.utterance):
            if len(token) > 2:
                terms[agent][token] += 1
    return {agent: counter.most_common(top_k) for agent, counter in terms.items()}

