from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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


def default_config() -> Config:
    current = Path(__file__).resolve()
    root_dir = current.parents[5] if len(current.parents) >= 6 else Path.cwd()
    llm_view_dir = current.parents[1]
    return Config(
        root_dir=root_dir,
        input_runs_dir=root_dir / "old_artifacts",
        topics_path=root_dir / "config" / "topics.json",
        output_analysis_dir=llm_view_dir / "llm_view_analysis_outputs",
        nrc_emotion_lexicon_path=llm_view_dir / "NRC-Emotion-Lexicon" / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt",
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

EVIDENCE_WORDS = {
    "evidence", "study", "studies", "data", "analysis", "report", "research", "source",
    "sources", "trial", "review", "finding", "findings", "statistic", "statistics",
    "dataset", "paper", "papers", "meta", "meta-analysis", "journal", "experiment",
}

REBUTTAL_WORDS = {
    "however", "but", "incorrect", "wrong", "misleading", "fails", "contradicts", "instead",
    "ignores", "actually", "despite", "although", "nevertheless", "inaccurate", "unsupported",
}

HEDGE_WORDS = {
    "may", "might", "could", "possibly", "perhaps", "suggest", "appears", "seems", "likely",
    "arguably", "potentially", "unclear", "approximately", "roughly", "plausible",
}

ASSERTIVE_WORDS = {
    "clearly", "definitely", "proves", "demonstrates", "certainly", "undeniably", "shows",
    "confirms", "obviously", "establishes", "conclusive", "without doubt",
}

STRONG_MODAL_WORDS = {
    "must", "will", "cannot", "always", "certainly", "definitely", "undeniably", "clearly",
}

WEAK_MODAL_WORDS = {
    "may", "might", "could", "can", "possibly", "perhaps", "suggests", "appears",
}

STANCE_PRO_WORDS = {
    "true", "real", "cover-up", "hidden", "suppressed", "censorship", "proof", "agenda",
}

STANCE_CON_WORDS = {
    "unsupported", "false", "misleading", "debunked", "evidence-based", "scientific", "inconsistent",
}

DEFAULT_NRC_EMOTIONS = [
    "anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust", "positive", "negative",
]

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
