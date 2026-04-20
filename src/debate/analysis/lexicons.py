"""Lightweight lexicons for language-use analysis.

These are intentionally small starter sets so the pipeline works out-of-the-box
without external resources. Teams can expand or replace these lists later.
"""

from __future__ import annotations

import os
from pathlib import Path
from collections import defaultdict
import re

# Consolidated Research Lexicons (Defaults)
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

def _load_emotion_lexicon() -> dict[str, set[str]]:
    """
    Attempt to load the full NRC Emotion Lexicon if available in the repository root.
    Otherwise, fall back to the lightweight starter set.
    """
    try:
        # Assuming we are running inside the repo, trace to the resources directory
        current_dir = Path(__file__).resolve().parent
        # Go up from src/debate/analysis -> src/debate/resources
        nrc_path = current_dir.parent / "resources" / "lexicons" / "NRC-Emotion-Lexicon" / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

        if nrc_path.exists():
            lex = defaultdict(set)
            with open(nrc_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = re.split(r"\s+", line)
                    if len(parts) == 3:
                        word, emotion, assoc = parts
                        if assoc == "1":
                            lex[emotion.lower()].add(word.lower())
            
            # The NRC lexicon contains negative, positive which aren't strictly emotions. 
            # We usually just use the 8 core emotions.
            core_emotions = {"anger", "fear", "trust", "disgust", "sadness", "joy", "anticipation", "surprise"}
            return {k: v for k, v in lex.items() if k in core_emotions}
            
    except Exception as e:
        pass
        
    return _BASIC_EMOTION_LEXICON

EMOTION_LEXICON: dict[str, set[str]] = _load_emotion_lexicon()

