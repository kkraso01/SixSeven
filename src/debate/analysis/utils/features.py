"""Feature extraction utilities for debate analysis.

Provides loaders for run artifacts (memory, transcript, metrics, config)
and helper functions for text analysis (keyword detection, safety flags).
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from textblob import TextBlob

from debate.core.schemas import MemoryState

from .lexicons import (
    MODALITY_STRONG_WORDS,
    MODALITY_WEAK_WORDS,
    UNCERTAINTY_WORDS,
)
from .lexicons import (
    load_nrc_emotion_lexicon as _lex_load_nrc_emotion_lexicon,
)
from .lexicons import (
    load_nrc_word_lexicon as _lex_load_nrc_word_lexicon,
)


@dataclass
class TranscriptRound:
    round_number: int
    claims: dict[str, str]
    recap_lines: list[str]


@dataclass
class TranscriptData:
    rounds: dict[int, TranscriptRound]


@dataclass
class MetricsData:
    civility: list[int]
    epistemic_quality: list[int]
    bridge_building: list[int]


@dataclass
class RunInputs:
    run_id: str
    memory: MemoryState
    transcript: TranscriptData
    metrics: MetricsData
    run_config: dict[str, object] | None


KEYWORD_PATTERN: re.Pattern[str] = re.compile(
    r"\b(strong point|concession|agreement|compelling|persuasive|valid point|acknowledge)\b",
    re.IGNORECASE,
)


PROFANITY_PATTERN: re.Pattern[str] = re.compile(
    r"\b(idiot|stupid|moron|dumb|shut up|liar|bullshit|nonsense)\b",
    re.IGNORECASE,
)

#: Maximum characters kept when excerpting a claim for reports.
EXCERPT_MAX_CHARS: int = 300
TOKEN_PATTERN: re.Pattern[str] = re.compile(r"[a-zA-Z']+")


def load_memory(run_dir: Path) -> MemoryState:
    memory_path = run_dir / "memory.json"
    return MemoryState.model_validate_json(memory_path.read_text(encoding="utf-8"))


def load_run_config(run_dir: Path) -> dict[str, object] | None:
    path = run_dir / "run_config.json"
    if not path.exists():
        return None
    result: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
    return result


def load_transcript(run_dir: Path) -> TranscriptData:
    path = run_dir / "transcript.md"
    if not path.exists():
        return TranscriptData(rounds={})
    text = path.read_text(encoding="utf-8")
    return parse_transcript(text)


def load_metrics(run_dir: Path, transcript: TranscriptData) -> MetricsData:
    metrics_path = run_dir / "metrics.csv"
    if metrics_path.exists():
        civility: list[int] = []
        epistemic: list[int] = []
        bridge: list[int] = []
        with metrics_path.open() as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                civility.append(int(row.get("civility", 0)))
                epistemic.append(int(row.get("epistemic_quality", 0)))
                bridge.append(int(row.get("bridge", 0)))
        return MetricsData(civility=civility, epistemic_quality=epistemic, bridge_building=bridge)
    return metrics_from_transcript(transcript)


def parse_transcript(text: str) -> TranscriptData:
    rounds: dict[int, TranscriptRound] = {}
    current_round: int | None = None
    current_speaker: str | None = None
    in_recap = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("## Round "):
            try:
                current_round = int(line.replace("## Round ", "").strip())
            except ValueError:
                current_round = None
            if current_round is not None:
                rounds[current_round] = TranscriptRound(
                    round_number=current_round,
                    claims={},
                    recap_lines=[],
                )
            current_speaker = None
            in_recap = False
            continue
        if line.startswith("### "):
            header = line.replace("### ", "").strip()
            if header == "Moderator Recap":
                in_recap = True
                current_speaker = None
            else:
                current_speaker = header
                in_recap = False
            continue
        if current_round is None:
            continue
        round_data = rounds[current_round]
        if line.startswith("Claim:") and current_speaker:
            round_data.claims[current_speaker] = line.replace("Claim:", "").strip()
        if in_recap and line:
            round_data.recap_lines.append(line)

    return TranscriptData(rounds=rounds)


def metrics_from_transcript(transcript: TranscriptData) -> MetricsData:
    civility: list[int] = []
    epistemic: list[int] = []
    bridge: list[int] = []
    for round_number in sorted(transcript.rounds):
        recap_lines = transcript.rounds[round_number].recap_lines
        civility.append(_extract_score(recap_lines, "Civility"))
        epistemic.append(_extract_score(recap_lines, "Epistemic Quality"))
        bridge.append(_extract_score(recap_lines, "Bridge Building"))
    return MetricsData(civility=civility, epistemic_quality=epistemic, bridge_building=bridge)


def _extract_score(lines: list[str], label: str) -> int:
    for line in lines:
        if line.startswith(f"{label}:"):
            value = line.replace(f"{label}:", "").strip()
            if value.isdigit():
                return int(value)
    return 0


def extract_recap_text(transcript: TranscriptData, round_number: int) -> str:
    round_data = transcript.rounds.get(round_number)
    if not round_data:
        return ""
    return " ".join(round_data.recap_lines)


def detect_recap_keyword(text: str) -> bool:
    return bool(KEYWORD_PATTERN.search(text))


def excerpt_for_round(
    transcript: TranscriptData,
    round_number: int,
    agent: str,
    fallback: str,
) -> str:
    round_data = transcript.rounds.get(round_number)
    if round_data and agent in round_data.claims:
        return round_data.claims[agent][:EXCERPT_MAX_CHARS]
    return fallback[:EXCERPT_MAX_CHARS]


def safety_flags_from_text(log_texts: list[tuple[int, str, str]]) -> list[str]:
    flags = []
    for round_number, speaker, text in log_texts:
        if PROFANITY_PATTERN.search(text):
            snippet = text.strip().split("\n")[0][:120]
            flags.append(f"Round {round_number} {speaker}: {snippet}")
    return flags


def analyze_utterance_features(
    text: str,
    uncertainty_lexicon: list[str] | None = None,
    strong_modality_lexicon: list[str] | None = None,
    weak_modality_lexicon: list[str] | None = None,
) -> dict[str, Any]:
    """Extract sentiment, polarity, subjectivity, and rhetorical scores."""
    text = str(text)
    tokens = re.findall(r"[a-zA-Z']+", text.lower())
    blob = TextBlob(text)

    # Use defaults if not provided
    uncertainty = uncertainty_lexicon or list(UNCERTAINTY_WORDS)
    strong = strong_modality_lexicon or list(MODALITY_STRONG_WORDS)
    weak = weak_modality_lexicon or list(MODALITY_WEAK_WORDS)

    return {
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity,
        "uncertainty_score": sum(t in uncertainty for t in tokens),
        "strong_modality_score": sum(t in strong for t in tokens),
        "weak_modality_score": sum(t in weak for t in tokens),
        "question_count": text.count("?"),
        "exclamation_count": text.count("!"),
        "word_count": len(tokens),
        "char_count": len(text),
    }


def tokenize_text(text: str) -> list[str]:
    """Tokenize plain text with the project's standard regex tokenizer."""
    return TOKEN_PATTERN.findall(str(text).lower())


def load_nrc_word_lexicon(path: Path) -> dict[str, set[str]]:
    """Compatibility wrapper around canonical lexicons loader."""
    return _lex_load_nrc_word_lexicon(path)


def extract_nrc_emotion_counts(
    text: str,
    nrc_word_lexicon: dict[str, set[str]],
    emotions: set[str] | list[str] | None = None,
    unique_tokens: bool = False,
) -> dict[str, int]:
    """Extract raw NRC emotion counts from text.

    Supports either:
    - word -> set(emotions)
    - emotion -> set(words)
    """
    known_emotions = {
        "anger",
        "anticipation",
        "disgust",
        "fear",
        "joy",
        "sadness",
        "surprise",
        "trust",
        "positive",
        "negative",
    }

    lexicon = nrc_word_lexicon
    if lexicon and set(lexicon.keys()).issubset(known_emotions):
        # Backward-compatible path for emotion -> words mappings.
        inverted: dict[str, set[str]] = defaultdict(set)
        for emotion, words in lexicon.items():
            for word in words:
                inverted[str(word).lower()].add(str(emotion).lower())
        lexicon = dict(inverted)

    tokens = tokenize_text(text)
    if unique_tokens:
        tokens = list(set(tokens))

    emotion_counts: Counter[str] = Counter()
    for token in tokens:
        for emotion in lexicon.get(token, set()):
            emotion_counts[emotion] += 1

    if emotions is None:
        ordered = sorted(emotion_counts.keys())
    else:
        ordered = sorted(set(emotions))

    return {emotion: int(emotion_counts.get(emotion, 0)) for emotion in ordered}


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


def load_nrc_emotion_lexicon(
    path: Path,
    emotions: set[str] | list[str] | None = None,
    nlp: Any | None = None,
    include_lemma_variants: bool = True,
) -> dict[str, set[str]]:
    """Compatibility wrapper around canonical lexicons loader."""
    return _lex_load_nrc_emotion_lexicon(
        path=path,
        emotions=emotions,
        nlp=nlp,
        include_lemma_variants=include_lemma_variants,
    )


def emotion_lexicon_scores(
    tokens: list[str], nrc_emotion_lexicon: dict[str, set[str]]
) -> dict[str, float]:
    """Compute normalized emotion overlap from token list and emotion -> words lexicon."""
    if not tokens:
        return {}

    token_set = set(tokens)
    denom = max(1, len(token_set))
    scores: dict[str, float] = {}
    for emotion, lexicon in nrc_emotion_lexicon.items():
        overlap = len(token_set & lexicon)
        scores[f"emotion_{emotion}"] = float(overlap) / float(denom)
    return scores


class EmotionAnalyzer:
    """Lazy-loaded BERT-based emotion classifier."""

    _instance = None
    _pipeline = None
    _model_name = None

    @classmethod
    def get_instance(cls, model_name: str | None = None) -> EmotionAnalyzer:
        if cls._instance is None:
            cls._instance = cls()
            from debate.core.config import DebateConfig

            config = DebateConfig.from_ini()
            cls._instance.set_model(config.adv_analysis_emotion_model)
        if model_name:
            cls._instance.set_model(model_name)
        return cls._instance

    def set_model(self, model_name: str) -> None:
        """Update model name. Note: only takes effect before pipeline is loaded."""
        if self._pipeline is None and model_name:
            self._model_name = model_name

    def _ensure_pipeline(self) -> None:
        if self._pipeline is None:
            from transformers import pipeline

            print(f"  Loading emotion model ({self._model_name})...")
            self._pipeline = pipeline(
                "text-classification",
                model=self._model_name,
                top_k=None,
            )

    def analyze(self, text: str) -> dict[str, float]:
        """Return dict of emotion scores (joy, sadness, anger, fear, love, surprise)."""
        self._ensure_pipeline()
        assert self._pipeline is not None
        try:
            # Type-cast results to avoid MyPy indexing errors on untyped pipeline output
            raw_results = self._pipeline(text)
            if not raw_results or not isinstance(raw_results, list):
                return {}

            results = raw_results[0]
            scores: dict[str, float] = {}
            for item in results:
                if isinstance(item, dict) and "label" in item and "score" in item:
                    label = str(item["label"]).lower().replace(" ", "_")
                    scores[f"emotion_{label}"] = float(item["score"])
            return scores
        except Exception:
            return {}


def load_run_inputs(run_dir: Path) -> RunInputs:
    memory = load_memory(run_dir)
    transcript = load_transcript(run_dir)
    metrics = load_metrics(run_dir, transcript)
    run_config = load_run_config(run_dir)
    return RunInputs(
        run_id=run_dir.name,
        memory=memory,
        transcript=transcript,
        metrics=metrics,
        run_config=run_config,
    )
