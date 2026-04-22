from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from debate.analysis.constants import DEFAULT_TRANSFORMER_EMOTION_MODEL
from debate.analysis.lexicons import (
    LLM_VIEW_ASSERTIVE_WORDS,
    LLM_VIEW_DEFAULT_NRC_EMOTIONS,
    LLM_VIEW_EVIDENCE_WORDS,
    LLM_VIEW_HEDGE_WORDS,
    NRC_EMOTION_LEXICON_PATH,
    LLM_VIEW_REBUTTAL_WORDS,
    LLM_VIEW_STANCE_CON_WORDS,
    LLM_VIEW_STANCE_PRO_WORDS,
    LLM_VIEW_STRONG_MODAL_WORDS,
    LLM_VIEW_WEAK_MODAL_WORDS,
)


@dataclass(frozen=True)
class Config:
    root_dir: Path
    input_runs_dir: Path
    topics_path: Path
    output_analysis_dir: Path
    max_runs: int | None = None
    overwrite_existing: bool = False
    analysis_schema_version: int = 3
    random_seed: int = 42
    bootstrap_samples: int = 1000
    top_n_terms: int = 25
    top_n_distinctive_terms: int = 30
    min_df_distinctive_terms: int = 2
    error_examples_per_type: int = 20
    turning_point_delta_threshold: float = 5.0
    nrc_emotion_lexicon_path: Path = Path()
    use_transformer_emotion_secondary: bool = False
    transformer_emotion_model: str = DEFAULT_TRANSFORMER_EMOTION_MODEL


def default_config() -> Config:
    current = Path(__file__).resolve()
    root_dir = current.parents[5] if len(current.parents) >= 6 else Path.cwd()
    llm_analysis_dir = current.parents[1]
    return Config(
        root_dir=root_dir,
        input_runs_dir=root_dir / "old_artifacts",
        topics_path=root_dir / "config" / "topics.json",
        output_analysis_dir=root_dir / "results" / "analysis" / "llm_analysis",
        nrc_emotion_lexicon_path=NRC_EMOTION_LEXICON_PATH,
    )


ROLE_TO_MODEL_KEY = {
    "proponent": "conspiracy",
    "opponent": "scientific",
    "moderator": "moderator",
}

ROLE_ALIASES = {
    "ca": "proponent",
    "conspiracy": "proponent",
    "conspiracy_advocate": "proponent",
    "conspiracy advocate": "proponent",
    "proponent": "proponent",
    "sa": "opponent",
    "scientific": "opponent",
    "scientific_advocate": "opponent",
    "scientific advocate": "opponent",
    "opponent": "opponent",
    "ma": "moderator",
    "moderator": "moderator",
}

ROLE_OPPONENT_MAP = {
    "proponent": "opponent",
    "opponent": "proponent",
}

EVIDENCE_WORDS = LLM_VIEW_EVIDENCE_WORDS
REBUTTAL_WORDS = LLM_VIEW_REBUTTAL_WORDS
HEDGE_WORDS = LLM_VIEW_HEDGE_WORDS
ASSERTIVE_WORDS = LLM_VIEW_ASSERTIVE_WORDS
STRONG_MODAL_WORDS = LLM_VIEW_STRONG_MODAL_WORDS
WEAK_MODAL_WORDS = LLM_VIEW_WEAK_MODAL_WORDS
STANCE_PRO_WORDS = LLM_VIEW_STANCE_PRO_WORDS
STANCE_CON_WORDS = LLM_VIEW_STANCE_CON_WORDS
DEFAULT_NRC_EMOTIONS = LLM_VIEW_DEFAULT_NRC_EMOTIONS

CORE_METRICS = [
    "confidence",
    "confidence_delta_within_role",
    "word_count",
    "avg_sentence_length",
    "lemma_type_token_ratio",
    "vader_compound",
    "evidence_density_per_1k_words",
    "rebuttal_density_per_1k_words",
    "hedge_density_per_1k_words",
    "assertive_density_per_1k_words",
    "strong_modal_density_per_1k_words",
    "weak_modal_density_per_1k_words",
    "moral_density_per_1k_words",
    "stance_proxy_score",
    "assertiveness_balance",
    "modality_balance",
    "lexical_overlap_prev_other_turn",
    "semantic_similarity_prev_other_turn",
    "lexical_overlap_prev_opponent_turn",
    "semantic_similarity_prev_opponent_turn",
    "novelty_vs_prev_turn_lexical",
    "novelty_vs_prev_turn_semantic",
    "novelty_vs_prev_opponent_turn_lexical",
    "novelty_vs_prev_opponent_turn_semantic",
]

BASELINE_METRICS = [
    "word_count",
    "type_token_ratio",
    "vader_compound",
    "citation_count",
    "question_count",
]

BINARY_RATE_COLS = ["has_evidence", "has_rebuttal", "has_question"]

METRICS_FOR_COMPARISON_PLOTS = [
    "confidence",
    "vader_compound",
    "evidence_density_per_1k_words",
    "rebuttal_density_per_1k_words",
    "hedge_density_per_1k_words",
    "assertive_density_per_1k_words",
    "moral_density_per_1k_words",
    "semantic_similarity_prev_other_turn",
]

METRICS_FOR_TRAJECTORY_PLOTS = [
    "confidence",
    "vader_compound",
    "evidence_density_per_1k_words",
    "hedge_density_per_1k_words",
    "assertive_density_per_1k_words",
    "semantic_similarity_prev_other_turn",
]
