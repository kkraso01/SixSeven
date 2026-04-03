"""Lightweight lexicons for language-use analysis.

These are intentionally small starter sets so the pipeline works out-of-the-box
without external resources. Teams can expand or replace these lists later.
"""

from __future__ import annotations

UNCERTAINTY_WORDS: set[str] = {
    "maybe",
    "may",
    "suggest",
    "suggests",
    "perhaps",
    "possibly",
    "likely",
    "unlikely",
    "appears",
    "seems",
    "could",
    "might",
    "unclear",
    "allegedly",
    "reportedly",
}

MODALITY_STRONG_WORDS: set[str] = {
    "must",
    "will",
    "cannot",
    "definitely",
    "certainly",
    "always",
    "prove",
    "proves",
    "proven",
    "undeniable",
}

MODALITY_WEAK_WORDS: set[str] = {
    "may",
    "might",
    "could",
    "can",
    "sometimes",
    "possibly",
    "arguably",
    "potentially",
}

MORAL_FOUNDATION_LEXICON: dict[str, set[str]] = {
    "care_harm": {"care", "harm", "hurt", "protect", "suffering", "safety", "risk"},
    "fairness_cheating": {"fair", "justice", "rights", "equal", "bias", "cheat", "fraud"},
    "loyalty_betrayal": {
        "loyal",
        "betray",
        "community",
        "nation",
        "patriot",
        "traitor",
    },
    "authority_subversion": {
        "authority",
        "expert",
        "institution",
        "law",
        "obedience",
        "corrupt",
    },
    "sanctity_degradation": {"pure", "sacred", "dirty", "contaminate", "unnatural", "toxic"},
}

EMOTION_LEXICON: dict[str, set[str]] = {
    "anger": {"angry", "rage", "furious", "outrage", "hostile"},
    "fear": {"fear", "afraid", "panic", "threat", "danger", "scared"},
    "trust": {"trust", "reliable", "credible", "evidence", "verified"},
    "disgust": {"disgust", "gross", "repulsive", "filthy"},
    "sadness": {"sad", "grief", "tragic", "loss"},
    "joy": {"hope", "joy", "relief", "encourage", "optimistic"},
    "anticipation": {"expect", "anticipate", "prepare", "forecast"},
    "surprise": {"surprised", "unexpected", "shocking", "astonishing"},
}

