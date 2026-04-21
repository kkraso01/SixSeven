"""Shared lexicons used across analysis implementations.

This module centralizes lexicon definitions for the different analysis pipelines
under ``src/debate/analysis`` while preserving legacy constant names.
"""

from __future__ import annotations

from typing import Any
from pathlib import Path
from collections import defaultdict
import re


ANALYSIS_DIR = Path(__file__).resolve().parent
DEBATE_SRC_DIR = ANALYSIS_DIR.parent
NRC_EMOTION_LEXICON_DIR = DEBATE_SRC_DIR / "resources" / "lexicons" / "NRC-Emotion-Lexicon"
NRC_EMOTION_LEXICON_PATH = NRC_EMOTION_LEXICON_DIR / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

# ---------------------------------------------------------------------------
# Base language-analysis lexicons (used by features.py / language_analysis.py)
# ---------------------------------------------------------------------------

UNCERTAINTY_WORDS: set[str] = {
    "maybe", "may", "suggest", "suggests", "perhaps", "possibly", "likely",
    "unlikely", "appears", "seems", "could", "might", "unclear", "allegedly",
    "reportedly", "potential", "concern", "concerns", "reasonable", "arguably", "uncertain"
}

MODALITY_STRONG_WORDS: set[str] = {
    "must", "will", "cannot", "definitely", "certainly", "always", "prove",
    "proves", "proven", "undeniable", "clearly", "demonstrates", "never",
    "obviously", "plainly", "shows", "confirms"
}

MODALITY_WEAK_WORDS: set[str] = {
    "may", "might", "could", "can", "sometimes", "possibly", "arguably",
    "potentially", "perhaps", "suggest", "suggests", "appears", "seems"
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


# ---------------------------------------------------------------------------
# Debate-analysis pipeline lexicons (debate_analysis/debate_analysis_pipeline.py)
# ---------------------------------------------------------------------------

DEBATE_ANALYSIS_NRC_TRUE_EMOTIONS: set[str] = {
    "anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"
}

DEBATE_ANALYSIS_STRONG_MODALITY_WORDS: set[str] = {
    "always", "must", "best", "clearly",
    "definitely", "definitively", "highest", "lowest",
    "never", "strongly", "unambiguously", "uncompromising",
    "undisputed", "undoubtedly", "unequivocal", "unequivocally",
    "unparalleled", "unsurpassed", "will"
}

DEBATE_ANALYSIS_WEAK_MODALITY_WORDS: set[str] = {
    "apparently", "appeared", "appearing", "appears",
    "conceivable", "could", "depend", "depended",
    "depending", "depends", "may", "maybe",
    "might", "nearly", "occasionally", "perhaps",
    "possible", "possibly", "seldom", "seldomly",
    "sometimes", "somewhat", "suggest", "suggests",
    "uncertain", "uncertainly"
}


# ---------------------------------------------------------------------------
# LLM-view lexicons (llm_analysis/analysis/config.py)
# ---------------------------------------------------------------------------

LLM_VIEW_EVIDENCE_WORDS: set[str] = {
    "evidence", "study", "studies", "data", "analysis", "report", "research", "source",
    "sources", "trial", "review", "finding", "findings", "statistic", "statistics",
    "dataset", "paper", "papers", "meta", "meta-analysis", "journal", "experiment",
}

LLM_VIEW_REBUTTAL_WORDS: set[str] = {
    "however", "but", "incorrect", "wrong", "misleading", "fails", "contradicts", "instead",
    "ignores", "actually", "despite", "although", "nevertheless", "inaccurate", "unsupported",
}

LLM_VIEW_HEDGE_WORDS: set[str] = {
    "may", "might", "could", "possibly", "perhaps", "suggest", "appears", "seems", "likely",
    "arguably", "potentially", "unclear", "approximately", "roughly", "plausible",
}

LLM_VIEW_ASSERTIVE_WORDS: set[str] = {
    "clearly", "definitely", "proves", "demonstrates", "certainly", "undeniably", "shows",
    "confirms", "obviously", "establishes", "conclusive", "without doubt",
}

LLM_VIEW_STRONG_MODAL_WORDS: set[str] = {
    "must", "will", "cannot", "always", "certainly", "definitely", "undeniably", "clearly",
}

LLM_VIEW_WEAK_MODAL_WORDS: set[str] = {
    "may", "might", "could", "can", "possibly", "perhaps", "suggests", "appears",
}

LLM_VIEW_STANCE_PRO_WORDS: set[str] = {
    "true", "real", "cover-up", "hidden", "suppressed", "censorship", "proof", "agenda",
}

LLM_VIEW_STANCE_CON_WORDS: set[str] = {
    "unsupported", "false", "misleading", "debunked", "evidence-based", "scientific", "inconsistent",
}

LLM_VIEW_DEFAULT_NRC_EMOTIONS: list[str] = [
    "anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust", "positive", "negative",
]

_BASIC_EMOTION_LEXICON: dict[str, set[str]] = {
    "anger": {"angry", "rage", "furious", "outrage", "hostile"},
    "fear": {"fear", "afraid", "panic", "threat", "danger", "scared"},
    "trust": {"trust", "reliable", "credible", "evidence", "verified"},
    "disgust": {"disgust", "gross", "repulsive", "filthy"},
    "sadness": {"sad", "grief", "tragic", "loss"},
    "joy": {"hope", "joy", "relief", "encourage", "optimistic"},
    "anticipation": {"expect", "anticipate", "prepare", "forecast"},
    "surprise": {"surprised", "unexpected", "shocking", "astonishing"},
}


def _parse_nrc_word_emotion_map(path: Path) -> dict[str, set[str]]:
    lex = defaultdict(set)
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            row = line.strip()
            if not row:
                continue
            parts = re.split(r"\s+", row)
            if len(parts) != 3:
                continue
            word, emotion, assoc = parts
            if assoc == "1":
                lex[word.lower()].add(emotion.lower())
    return dict(lex)


def _normalize_lex_token(token: Any) -> str:
    value = str(token).strip().lower()
    value = re.sub(r"\s+", " ", value)
    return value


def _lemma_variants(nlp: Any, term: str) -> set[str]:
    variants: set[str] = set()
    if not term:
        return variants
    variants.add(term)
    try:
        doc = nlp(term)
    except Exception:
        doc = None
    if doc is not None:
        lemmas = [tok.lemma_.lower() for tok in doc if tok.is_alpha and tok.lemma_]
        if lemmas:
            variants.add(" ".join(lemmas))
            if len(lemmas) == 1:
                variants.add(lemmas[0])
    return {variant for variant in variants if variant}


def load_nrc_word_lexicon(path: Path | None = None) -> dict[str, set[str]]:
    """Load NRC lexicon as word -> set(emotions) from shared resources by default."""
    selected = path or NRC_EMOTION_LEXICON_PATH
    if not selected.exists():
        raise FileNotFoundError(f"NRC lexicon not found at: {selected}")
    return _parse_nrc_word_emotion_map(selected)


def load_nrc_emotion_lexicon(
    path: Path | None = None,
    emotions: set[str] | list[str] | None = None,
    nlp: Any | None = None,
    include_lemma_variants: bool = True,
) -> dict[str, set[str]]:
    """Load NRC lexicon as emotion -> set(words), optionally with lemma variants."""
    word_to_emotions = load_nrc_word_lexicon(path)

    if emotions is None:
        emotion_keys = sorted({emotion for emos in word_to_emotions.values() for emotion in emos})
    else:
        emotion_keys = sorted(set(str(emotion).lower() for emotion in emotions))

    emotion_lexicon: dict[str, set[str]] = {emotion: set() for emotion in emotion_keys}
    for word, word_emotions in word_to_emotions.items():
        normalized_word = _normalize_lex_token(word)
        if include_lemma_variants and nlp is not None:
            variants = _lemma_variants(nlp, normalized_word)
        else:
            variants = {normalized_word}

        for emotion in word_emotions:
            if emotion in emotion_lexicon:
                emotion_lexicon[emotion].update(variants)

    return emotion_lexicon

def _load_emotion_lexicon() -> dict[str, set[str]]:
    """
    Attempt to load the full NRC Emotion Lexicon from the shared resources folder.
    Otherwise, fall back to the lightweight starter set.
    """
    try:
        if NRC_EMOTION_LEXICON_PATH.exists():
            core_emotions = {"anger", "fear", "trust", "disgust", "sadness", "joy", "anticipation", "surprise"}
            return load_nrc_emotion_lexicon(path=NRC_EMOTION_LEXICON_PATH, emotions=core_emotions, include_lemma_variants=False)
            
    except Exception as e:
        pass
        
    return _BASIC_EMOTION_LEXICON

EMOTION_LEXICON: dict[str, set[str]] = _load_emotion_lexicon()


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

NRC_TRUE_EMOTIONS: set[str] = DEBATE_ANALYSIS_NRC_TRUE_EMOTIONS
STRONG_MODALITY_WORDS: set[str] = DEBATE_ANALYSIS_STRONG_MODALITY_WORDS
WEAK_MODALITY_WORDS: set[str] = DEBATE_ANALYSIS_WEAK_MODALITY_WORDS

