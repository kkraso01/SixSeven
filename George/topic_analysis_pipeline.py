#!/usr/bin/env python3
"""
Topic-level analysis pipeline for the SixSeven agentic debate project.

What it does
------------
1) Reads all debate runs from old_artifacts/ (or old_artifacts.zip).
2) Groups runs by unique topic_id from experiment_metadata.json.
3) Estimates who usually wins per topic.
4) Runs BERT-based emotion analysis on the utterance column.
5) Computes language-use features per topic and per role (conspiracy/scientific):
   - top words and bigrams
   - uncertainty lexicon usage
   - strong / weak modality usage
   - moral framing (eMFD-style dictionary scores)
   - optional NRC lexical emotion profile

Example
-------
  python topic_analysis_pipeline.py \
  --artifacts /path/to/old_artifacts \
  --output_dir topic_analysis_results \
  --uncertainty_lexicon /path/to/uncertainty.txt \
  --strong_modality_lexicon /path/to/strong_modals.txt \
  --weak_modality_lexicon /path/to/weak_modals.txt \
  --nrc_lexicon /path/to/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt \
  --emfd_lexicon /path/to/emfd.csv
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Optional heavy deps are imported lazily inside functions.


# -----------------------------
# General utilities
# -----------------------------

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same",
    "so", "than", "too", "very", "can", "will", "just", "don", "should", "now", "is", "are",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "this",
    "that", "these", "those", "it", "its", "as", "i", "you", "he", "she", "they", "them", "we",
    "our", "ours", "your", "yours", "his", "her", "hers", "their", "theirs", "me", "my", "mine",
    "us", "what", "which", "who", "whom", "am", "because", "until", "ll", "re", "ve", "m", "s",
}

ROLE_MAP = {
    "ca": "conspiracy",
    "conspiracy": "conspiracy",
    "conspiracy_agent": "conspiracy",
    "proponent": "conspiracy",
    "scientific": "scientist",
    "science": "scientist",
    "sa": "scientist",
    "scientist": "scientist",
    "scientific_agent": "scientist",
    "opponent": "scientist",
    "moderator": "moderator",
    "mod": "moderator",
}


def normalize_role(role: str) -> str:
    if role is None or (isinstance(role, float) and math.isnan(role)):
        return "unknown"
    role = str(role).strip().lower()
    return ROLE_MAP.get(role, role)


def clean_text(text: str) -> str:
    text = str(text or "")
    text = text.replace("\n", " ")
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^A-Za-z\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def tokenize(text: str) -> List[str]:
    text = clean_text(text)
    toks = [t for t in text.split() if len(t) > 2 and t not in STOPWORDS]
    return toks


def make_bigrams(tokens: Sequence[str]) -> List[str]:
    return [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]


def safe_json_load(path_or_bytes) -> dict:
    if hasattr(path_or_bytes, "read"):
        return json.load(path_or_bytes)
    with open(path_or_bytes, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------
# Data loading
# -----------------------------

@dataclass
class RunRecord:
    run_name: str
    topic_id: str
    topic_category: Optional[str]
    topic_description: Optional[str]
    model_config: Optional[str]
    debate_log: pd.DataFrame
    final_report: dict
    metadata: dict


class ArtifactReader:
    def __init__(self, artifacts_path: str):
        self.artifacts_path = Path(artifacts_path)
        if not self.artifacts_path.exists():
            raise FileNotFoundError(f"Artifacts path not found: {self.artifacts_path}")
        self.is_zip = self.artifacts_path.suffix.lower() == ".zip"
        self._zip = zipfile.ZipFile(self.artifacts_path) if self.is_zip else None

        # Support both:
        #   /path/to/old_artifacts/
        # and
        #   /path/to/parent_containing_old_artifacts/
        self.base_dir = self.artifacts_path
        if not self.is_zip and not list(self.base_dir.glob("run_*")):
            nested = self.base_dir / "old_artifacts"
            if nested.exists() and nested.is_dir():
                self.base_dir = nested

    def list_run_names(self) -> List[str]:
        if self.is_zip:
            runs = set()
            for name in self._zip.namelist():
                m = re.match(r"(.*/run_[^/]+)/", name)
                if m:
                    runs.add(m.group(1))
            return sorted(runs)

        return sorted([str(p) for p in self.base_dir.glob("run_*") if p.is_dir()])

    def _open_text(self, path: str):
        if self.is_zip:
            return io.TextIOWrapper(self._zip.open(path), encoding="utf-8")
        return open(path, "r", encoding="utf-8")

    def _read_csv(self, path: str) -> pd.DataFrame:
        if self.is_zip:
            return pd.read_csv(self._zip.open(path))
        return pd.read_csv(path)

    def load_run(self, run_name: str) -> RunRecord:
        meta_path = f"{run_name}/experiment_metadata.json"
        debate_path = f"{run_name}/debate_log.csv"
        final_path = f"{run_name}/final_report.json"

        with self._open_text(meta_path) as f:
            metadata = json.load(f)
        debate_log = self._read_csv(debate_path)
        with self._open_text(final_path) as f:
            final_report = json.load(f)

        if "speaker_role" in debate_log.columns:
            debate_log["speaker_role_normalized"] = debate_log["speaker_role"].map(normalize_role)
        else:
            debate_log["speaker_role_normalized"] = "unknown"

        return RunRecord(
            run_name=os.path.basename(run_name),
            topic_id=metadata.get("topic_id", "unknown_topic"),
            topic_category=metadata.get("topic_category"),
            topic_description=metadata.get("topic_description"),
            model_config=metadata.get("model_config"),
            debate_log=debate_log,
            final_report=final_report,
            metadata=metadata,
        )


# -----------------------------
# Winner inference
# -----------------------------

def infer_winner_from_final_report(final_report: dict) -> Tuple[str, str]:
    """
    Returns (winner, method).

    Winner is one of: scientist, conspiracy, tie, unknown
    Method describes which rule was used.
    """
    traj = final_report.get("stance_trajectory", {}) or {}
    if isinstance(traj, dict):
        # Most common keys seen in your data are SA and CA.
        sa = traj.get("SA") or traj.get("sa") or traj.get("scientist") or traj.get("scientific")
        ca = traj.get("CA") or traj.get("ca") or traj.get("conspiracy") or traj.get("proponent")
        if isinstance(sa, list) and isinstance(ca, list) and sa and ca:
            sa_final = sa[-1]
            ca_final = ca[-1]
            if sa_final > ca_final:
                return "scientist", "stance_trajectory_final"
            if ca_final > sa_final:
                return "conspiracy", "stance_trajectory_final"
            return "tie", "stance_trajectory_final"

    summary = str(final_report.get("outcome_summary", "")).lower()
    if summary:
        scientist_patterns = [
            r"\bsa successfully\b",
            r"\bscientific\b.*\bwon\b",
            r"\bscientist\b.*\bwon\b",
            r"\bopponent\b.*\bwon\b",
            r"\bstrengthen(?:ed|ing)?\s+sa\b",
            r"\btowards\s+their\s+stance\b",
        ]
        conspiracy_patterns = [
            r"\bca successfully\b",
            r"\bconspiracy\b.*\bwon\b",
            r"\bproponent\b.*\bwon\b",
            r"\bconspiracist\b.*\bwon\b",
        ]
        for pat in scientist_patterns:
            if re.search(pat, summary):
                return "scientist", "outcome_summary_regex"
        for pat in conspiracy_patterns:
            if re.search(pat, summary):
                return "conspiracy", "outcome_summary_regex"

    return "unknown", "none"


# -----------------------------
# Lexicon loading and scoring
# -----------------------------

def load_simple_lexicon(path: Optional[str]) -> set:
    """Load one item per line. Ignores empty lines and comment lines."""
    if not path or not Path(path).exists():
        return set()
    items = set()
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip().lower()
            if not line or line.startswith("#"):
                continue
            items.add(line)
    return items


def score_lexicon_terms(tokens: Sequence[str], lexicon: set) -> Tuple[int, float]:
    if not tokens:
        return 0, 0.0
    count = sum(1 for t in tokens if t in lexicon)
    return count, count / len(tokens)


def load_nrc_emotion_lexicon(path: Optional[str]) -> Dict[str, set]:
    """
    Expects the standard NRC format: word<TAB>emotion<TAB>association
    Returns: emotion -> set(words)
    """
    emotions = defaultdict(set)
    if not path or not Path(path).exists():
        return emotions

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            word, emotion, assoc = parts
            if assoc == "1":
                emotions[emotion.lower()].add(word.lower())
    return emotions


def score_nrc_emotions(tokens: Sequence[str], emotion_lex: Dict[str, set]) -> Dict[str, float]:
    if not tokens:
        return {emo: 0.0 for emo in emotion_lex.keys()}
    result = {}
    for emo, words in emotion_lex.items():
        count = sum(1 for t in tokens if t in words)
        result[emo] = count / len(tokens)
    return result


def load_emfd_lexicon(path: Optional[str]) -> Optional[pd.DataFrame]:
    """
    Tries to load a local eMFD-style CSV/TSV file.

    The loader is intentionally flexible because different downloads / exports
    may use slightly different column names.

    Expected minimum:
      - one column containing the token / word / term
      - several numeric columns for moral dimensions, such as:
        care, fairness, loyalty, authority, purity
        and optionally virtue/vice versions

    Returns the raw DataFrame or None if unavailable.
    """
    if not path or not Path(path).exists():
        return None

    suffix = Path(path).suffix.lower()
    sep = "\t" if suffix in {".tsv", ".txt"} else ","
    df = pd.read_csv(path, sep=sep)

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


def score_emfd_tokens(tokens: Sequence[str], emfd_df: Optional[pd.DataFrame]) -> Dict[str, float]:
    """
    Returns average moral scores over the tokens that matched the dictionary.
    If no eMFD file is provided, returns {}.
    """
    if emfd_df is None or not tokens:
        return {}

    token_col_candidates = ["word", "term", "token", "lemma", "feature"]
    token_col = next((c for c in token_col_candidates if c in emfd_df.columns), None)
    if token_col is None:
        return {}

    numeric_cols = [c for c in emfd_df.columns if c != token_col and pd.api.types.is_numeric_dtype(emfd_df[c])]
    if not numeric_cols:
        return {}

    lookup = emfd_df.set_index(token_col)
    matched = [t for t in tokens if t in lookup.index]
    if not matched:
        return {c: 0.0 for c in numeric_cols}

    subset = lookup.loc[matched, numeric_cols]
    if isinstance(subset, pd.Series):
        subset = subset.to_frame().T
    return subset.mean(axis=0).to_dict()


# -----------------------------
# BERT emotion analysis
# -----------------------------

def run_bert_emotion_classifier(
    texts: Sequence[str],
    model_name: str = "j-hartmann/emotion-english-distilroberta-base",
    batch_size: int = 16,
    max_length: int = 256,
    device: Optional[int] = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per text:
      predicted_emotion, plus probability columns for each label.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import torch

    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    if device >= 0:
        model = model.to(device)
    model.eval()

    labels = [model.config.id2label[i] for i in sorted(model.config.id2label)]
    all_rows = []

    for start in range(0, len(texts), batch_size):
        batch_texts = [str(t or "")[:2000] for t in texts[start : start + batch_size]]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if device >= 0:
            enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()

        pred_ids = probs.argmax(axis=1)
        for i in range(len(batch_texts)):
            row = {labels[j]: float(probs[i, j]) for j in range(len(labels))}
            row["predicted_emotion"] = labels[pred_ids[i]]
            all_rows.append(row)

    return pd.DataFrame(all_rows)


# -----------------------------
# Topic-level aggregation
# -----------------------------

def top_k_counter(items: Iterable[str], k: int = 20) -> List[Tuple[str, int]]:
    return Counter(items).most_common(k)


def flatten_counter_list(counter_list: List[Tuple[str, int]]) -> str:
    return ", ".join([f"{token}:{count}" for token, count in counter_list])


def build_master_utterance_table(run_records: Sequence[RunRecord]) -> pd.DataFrame:
    rows = []
    for run in run_records:
        df = run.debate_log.copy()
        if "utterance" not in df.columns:
            continue
        for _, r in df.iterrows():
            rows.append(
                {
                    "run_name": run.run_name,
                    "topic_id": run.topic_id,
                    "topic_category": run.topic_category,
                    "topic_description": run.topic_description,
                    "model_config": run.model_config,
                    "speaker_role": r.get("speaker_role"),
                    "role": r.get("speaker_role_normalized", "unknown"),
                    "round": r.get("round"),
                    "utterance": str(r.get("utterance", "")),
                    "stance": r.get("stance"),
                    "confidence": r.get("confidence"),
                }
            )
    return pd.DataFrame(rows)


def aggregate_winners(run_records: Sequence[RunRecord]) -> pd.DataFrame:
    rows = []
    for run in run_records:
        winner, method = infer_winner_from_final_report(run.final_report)
        rows.append(
            {
                "run_name": run.run_name,
                "topic_id": run.topic_id,
                "topic_category": run.topic_category,
                "model_config": run.model_config,
                "winner": winner,
                "winner_inference_method": method,
            }
        )
    per_run = pd.DataFrame(rows)
    if per_run.empty:
        return per_run

    summary_rows = []
    for topic_id, g in per_run.groupby("topic_id"):
        counts = g["winner"].value_counts().to_dict()
        dominant = g["winner"].mode().iloc[0] if not g["winner"].mode().empty else "unknown"
        summary_rows.append(
            {
                "topic_id": topic_id,
                "num_debates": len(g),
                "scientist_wins": counts.get("scientist", 0),
                "conspiracy_wins": counts.get("conspiracy", 0),
                "ties": counts.get("tie", 0),
                "unknown": counts.get("unknown", 0),
                "most_likely_winner": dominant,
                "scientist_win_rate": counts.get("scientist", 0) / len(g),
                "conspiracy_win_rate": counts.get("conspiracy", 0) / len(g),
            }
        )
    return per_run, pd.DataFrame(summary_rows).sort_values("topic_id")


def aggregate_topic_role_language(
    utter_df: pd.DataFrame,
    uncertainty_lex: set,
    strong_modality_lex: set,
    weak_modality_lex: set,
    nrc_emotion_lex: Dict[str, set],
    emfd_df: Optional[pd.DataFrame],
    top_k_words: int = 25,
    top_k_bigrams: int = 20,
) -> pd.DataFrame:
    rows = []

    filtered = utter_df[utter_df["role"].isin(["conspiracy", "scientist"])].copy()
    if filtered.empty:
        return pd.DataFrame()

    for (topic_id, role), g in filtered.groupby(["topic_id", "role"]):
        texts = g["utterance"].fillna("").astype(str).tolist()
        tokens = []
        for txt in texts:
            tokens.extend(tokenize(txt))
        bigrams = make_bigrams(tokens)

        uncertainty_count, uncertainty_rate = score_lexicon_terms(tokens, uncertainty_lex)
        strong_count, strong_rate = score_lexicon_terms(tokens, strong_modality_lex)
        weak_count, weak_rate = score_lexicon_terms(tokens, weak_modality_lex)
        nrc_scores = score_nrc_emotions(tokens, nrc_emotion_lex)
        emfd_scores = score_emfd_tokens(tokens, emfd_df)

        row = {
            "topic_id": topic_id,
            "role": role,
            "num_utterances": len(texts),
            "num_tokens": len(tokens),
            "uncertainty_count": uncertainty_count,
            "uncertainty_rate": uncertainty_rate,
            "strong_modality_count": strong_count,
            "strong_modality_rate": strong_rate,
            "weak_modality_count": weak_count,
            "weak_modality_rate": weak_rate,
            "top_words": flatten_counter_list(top_k_counter(tokens, top_k_words)),
            "top_bigrams": flatten_counter_list(top_k_counter(bigrams, top_k_bigrams)),
        }
        row.update({f"nrc_{k}": v for k, v in sorted(nrc_scores.items())})
        row.update({f"emfd_{k}": v for k, v in sorted(emfd_scores.items())})
        rows.append(row)

    return pd.DataFrame(rows).sort_values(["topic_id", "role"])


def aggregate_topic_emotions_with_bert(
    utter_df: pd.DataFrame,
    model_name: str,
    batch_size: int,
    max_length: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if utter_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    emotion_df = run_bert_emotion_classifier(
        utter_df["utterance"].fillna("").astype(str).tolist(),
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
    )

    full = pd.concat([utter_df.reset_index(drop=True), emotion_df.reset_index(drop=True)], axis=1)

    label_cols = [c for c in emotion_df.columns if c != "predicted_emotion"]
    topic_rows = []
    for topic_id, g in full.groupby("topic_id"):
        pred_counts = g["predicted_emotion"].value_counts(normalize=True).to_dict()
        avg_probs = g[label_cols].mean(axis=0).to_dict()
        top_label = g["predicted_emotion"].mode().iloc[0] if not g["predicted_emotion"].mode().empty else None
        row = {
            "topic_id": topic_id,
            "num_utterances": len(g),
            "dominant_bert_emotion": top_label,
        }
        row.update({f"bert_share_{k}": v for k, v in sorted(pred_counts.items())})
        row.update({f"bert_meanprob_{k}": v for k, v in sorted(avg_probs.items())})
        topic_rows.append(row)

    return full, pd.DataFrame(topic_rows).sort_values("topic_id")


# -----------------------------
# Main pipeline
# -----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Topic-level analysis pipeline for debate runs")
    parser.add_argument("--artifacts", type=str, required=True, help="Path to old_artifacts directory or old_artifacts.zip")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory where outputs will be written")

    parser.add_argument("--uncertainty_lexicon", type=str, default=None)
    parser.add_argument("--strong_modality_lexicon", type=str, default=None)
    parser.add_argument("--weak_modality_lexicon", type=str, default=None)
    parser.add_argument("--nrc_lexicon", type=str, default=None)
    parser.add_argument("--emfd_lexicon", type=str, default=None)

    parser.add_argument("--emotion_model", type=str, default="j-hartmann/emotion-english-distilroberta-base")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    reader = ArtifactReader(args.artifacts)
    run_names = reader.list_run_names()
    print(f"Found {len(run_names)} runs")

    run_records = []
    for run_name in run_names:
        try:
            run_records.append(reader.load_run(run_name))
        except Exception as e:
            print(f"[WARN] Skipping {run_name}: {e}")

    if not run_records:
        raise RuntimeError("No valid run folders were loaded.")

    # Save inventory of runs
    run_inventory = pd.DataFrame(
        [
            {
                "run_name": r.run_name,
                "topic_id": r.topic_id,
                "topic_category": r.topic_category,
                "topic_description": r.topic_description,
                "model_config": r.model_config,
            }
            for r in run_records
        ]
    )
    run_inventory.to_csv(out_dir / "run_inventory.csv", index=False)

    # Master utterance table
    utter_df = build_master_utterance_table(run_records)
    utter_df.to_csv(out_dir / "all_utterances_with_metadata.csv", index=False)

    # Winner analysis
    per_run_winners, per_topic_winners = aggregate_winners(run_records)
    per_run_winners.to_csv(out_dir / "winner_per_run.csv", index=False)
    per_topic_winners.to_csv(out_dir / "winner_per_topic.csv", index=False)

    # Lexicons
    uncertainty_lex = load_simple_lexicon(args.uncertainty_lexicon)
    strong_modality_lex = load_simple_lexicon(args.strong_modality_lexicon)
    weak_modality_lex = load_simple_lexicon(args.weak_modality_lexicon)
    nrc_emotion_lex = load_nrc_emotion_lexicon(args.nrc_lexicon)
    emfd_df = load_emfd_lexicon(args.emfd_lexicon)

    # Language analysis (topic x role)
    topic_role_language = aggregate_topic_role_language(
        utter_df,
        uncertainty_lex,
        strong_modality_lex,
        weak_modality_lex,
        nrc_emotion_lex,
        emfd_df,
    )
    topic_role_language.to_csv(out_dir / "language_by_topic_and_role.csv", index=False)

    # BERT emotions
    utter_emotions, topic_emotions = aggregate_topic_emotions_with_bert(
        utter_df=utter_df,
        model_name=args.emotion_model,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    utter_emotions.to_csv(out_dir / "utterance_level_bert_emotions.csv", index=False)
    topic_emotions.to_csv(out_dir / "bert_emotions_per_topic.csv", index=False)

    # Final joined summary for easy reporting
    topic_summary = per_topic_winners.merge(topic_emotions, on="topic_id", how="outer")
    topic_summary.to_csv(out_dir / "topic_summary_combined.csv", index=False)

    print("Done. Files written to:", out_dir)
    for p in sorted(out_dir.glob("*.csv")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
