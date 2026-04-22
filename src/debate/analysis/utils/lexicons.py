"""Shared lexicons used across analysis implementations.

This module centralizes lexicon definitions for the different analysis pipelines
under ``src/debate/analysis`` while preserving legacy constant names.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any

UTILS_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = UTILS_DIR.parent
DEBATE_SRC_DIR = ANALYSIS_DIR.parent
NRC_EMOTION_LEXICON_DIR = DEBATE_SRC_DIR / "resources" / "lexicons" / "NRC-Emotion-Lexicon"
NRC_EMOTION_LEXICON_PATH = NRC_EMOTION_LEXICON_DIR / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"
EMFD_LEXICON_PATH = DEBATE_SRC_DIR / "resources" / "lexicons" / "eMFD" / "emfd_scoring.csv"

BUILTINS_JSON_PATH = DEBATE_SRC_DIR / "resources" / "lexicons" / "builtins.json"

import json

with open(BUILTINS_JSON_PATH, encoding="utf-8") as _f:
    _builtins = json.load(_f)

# ---------------------------------------------------------------------------
# Base language-analysis lexicons
# ---------------------------------------------------------------------------

UNCERTAINTY_WORDS: set[str] = set(_builtins["UNCERTAINTY_WORDS"])
MODALITY_STRONG_WORDS: set[str] = set(_builtins["MODALITY_STRONG_WORDS"])
MODALITY_WEAK_WORDS: set[str] = set(_builtins["MODALITY_WEAK_WORDS"])
MORAL_FOUNDATION_LEXICON: dict[str, set[str]] = {
    k: set(v) for k, v in _builtins["MORAL_FOUNDATION_LEXICON"].items()
}


# ---------------------------------------------------------------------------
# Topic-analysis pipeline lexicons
# ---------------------------------------------------------------------------

TOPIC_ANALYSIS_UNCERTAINTY_WORDS: set[str] = set(_builtins["TOPIC_ANALYSIS_UNCERTAINTY_WORDS"])
TOPIC_ANALYSIS_STRONG_MODALITY_WORDS: set[str] = set(
    _builtins["TOPIC_ANALYSIS_STRONG_MODALITY_WORDS"]
)
TOPIC_ANALYSIS_WEAK_MODALITY_WORDS: set[str] = set(_builtins["TOPIC_ANALYSIS_WEAK_MODALITY_WORDS"])


# ---------------------------------------------------------------------------
# Debate-analysis pipeline lexicons
# ---------------------------------------------------------------------------

DEBATE_ANALYSIS_NRC_TRUE_EMOTIONS: set[str] = set(_builtins["DEBATE_ANALYSIS_NRC_TRUE_EMOTIONS"])
DEBATE_ANALYSIS_STRONG_MODALITY_WORDS: set[str] = set(
    _builtins["DEBATE_ANALYSIS_STRONG_MODALITY_WORDS"]
)
DEBATE_ANALYSIS_WEAK_MODALITY_WORDS: set[str] = set(
    _builtins["DEBATE_ANALYSIS_WEAK_MODALITY_WORDS"]
)


# ---------------------------------------------------------------------------
# LLM-view lexicons
# ---------------------------------------------------------------------------

LLM_VIEW_EVIDENCE_WORDS: set[str] = set(_builtins["LLM_VIEW_EVIDENCE_WORDS"])
LLM_VIEW_REBUTTAL_WORDS: set[str] = set(_builtins["LLM_VIEW_REBUTTAL_WORDS"])
LLM_VIEW_HEDGE_WORDS: set[str] = set(_builtins["LLM_VIEW_HEDGE_WORDS"])
LLM_VIEW_ASSERTIVE_WORDS: set[str] = set(_builtins["LLM_VIEW_ASSERTIVE_WORDS"])
LLM_VIEW_STRONG_MODAL_WORDS: set[str] = set(_builtins["LLM_VIEW_STRONG_MODAL_WORDS"])
LLM_VIEW_WEAK_MODAL_WORDS: set[str] = set(_builtins["LLM_VIEW_WEAK_MODAL_WORDS"])
LLM_VIEW_STANCE_PRO_WORDS: set[str] = set(_builtins["LLM_VIEW_STANCE_PRO_WORDS"])
LLM_VIEW_STANCE_CON_WORDS: set[str] = set(_builtins["LLM_VIEW_STANCE_CON_WORDS"])
LLM_VIEW_DEFAULT_NRC_EMOTIONS: list[str] = _builtins["LLM_VIEW_DEFAULT_NRC_EMOTIONS"]
_BASIC_EMOTION_LEXICON: dict[str, set[str]] = {
    k: set(v) for k, v in _builtins["_BASIC_EMOTION_LEXICON"].items()
}


def _parse_nrc_word_emotion_map(path: Path) -> dict[str, set[str]]:
    lex = defaultdict(set)
    with open(path, encoding="utf-8") as handle:
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


def load_emfd_lexicon(path: Path | str | None = None) -> Any | None:
    """Load an eMFD-style CSV/TSV file from the shared path by default."""
    selected = Path(path) if path is not None else EMFD_LEXICON_PATH
    if not selected.exists():
        return None

    import pandas as pd

    suffix = selected.suffix.lower()
    sep = "\t" if suffix in {".tsv", ".txt"} else ","
    df = pd.read_csv(selected, sep=sep)

    lower_map = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=lower_map)

    token_col_candidates = ["word", "term", "token", "lemma", "feature"]
    token_col = next((c for c in token_col_candidates if c in df.columns), None)
    if token_col is None:
        raise ValueError(
            "Could not detect the token column in the eMFD file. "
            "Please rename it to one of: word, term, token, lemma, feature."
        )

    df[token_col] = df[token_col].astype(str).str.lower().str.strip()
    return df


def _load_emotion_lexicon() -> dict[str, set[str]]:
    """
    Attempt to load the full NRC Emotion Lexicon from the shared resources folder.
    Otherwise, fall back to the lightweight starter set.
    """
    try:
        if NRC_EMOTION_LEXICON_PATH.exists():
            core_emotions = {
                "anger",
                "fear",
                "trust",
                "disgust",
                "sadness",
                "joy",
                "anticipation",
                "surprise",
            }
            return load_nrc_emotion_lexicon(
                path=NRC_EMOTION_LEXICON_PATH, emotions=core_emotions, include_lemma_variants=False
            )

    except Exception:
        pass

    return _BASIC_EMOTION_LEXICON


EMOTION_LEXICON: dict[str, set[str]] = _load_emotion_lexicon()


# ---------------------------------------------------------------------------
# Backward-compatible aliases
# ---------------------------------------------------------------------------

NRC_TRUE_EMOTIONS: set[str] = DEBATE_ANALYSIS_NRC_TRUE_EMOTIONS
STRONG_MODALITY_WORDS: set[str] = DEBATE_ANALYSIS_STRONG_MODALITY_WORDS
WEAK_MODALITY_WORDS: set[str] = DEBATE_ANALYSIS_WEAK_MODALITY_WORDS
