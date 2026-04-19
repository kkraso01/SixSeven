import re
import json
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from transformers import pipeline


# =========================================================
# CONFIG
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent.parent.parent.parent


INPUT_RUNS_DIR = REPO_ROOT / "old_artifacts"
OUTPUT_ANALYSIS_DIR = BASE_DIR / "debate_analysis_outputs"

NRC_PATH = REPO_ROOT / "lexicons" / "NRC-Emotion-Lexicon" / "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"

OVERWRITE_EXISTING = True
EMOTION_MODEL_NAME = "bhadresh-savani/bert-base-uncased-emotion"

NRC_TRUE_EMOTIONS = {
    "anger", "anticipation", "disgust", "fear",
    "joy", "sadness", "surprise", "trust"
}

# Strong / weak modality words
# generic
# STRONG_MODALITY_WORDS = {
#     "must", "clearly", "definitely", "certainly", "undeniably",
#     "always", "never", "prove", "proves", "proved",
#     "demonstrate", "demonstrates", "demonstrated",
#     "show", "shows", "showed", "cannot", "can't",
#     "will", "obviously", "plainly"
# }


# These are given in the project PDF
STRONG_MODALITY_WORDS = {
    "always", "must", "best", "clearly",
    "definitely", "definitively", "hgihest", "lowest",
    "never", "strongly", "unambiguously", "uncompromising",
    "undisputed", "undoubtedly", "unequivocal", "unequivocally",
    "unparalleled", "unsurpassed", "will"
}

# generic
# WEAK_MODALITY_WORDS = {
#     "may", "might", "could", "can", "perhaps", "possibly",
#     "seems", "seem", "appear", "appears", "appeared",
#     "suggest", "suggests", "suggested", "likely", "unlikely",
#     "arguably", "maybe", "potentially", "apparently"
# }

# from pdf
WEAK_MODALITY_WORDS = {
    "apparently", "appeared", "appearing", "appears",
    "conceivable", "could", "depend", "depended",
    "depending", "depends", "may", "maybe",
    "might", "nearly", "occasionally", "perhaps",
    "possible", "possibly", "seldom", "seldomly",
    "sometimes", "somewhat", "suggest", "suggests",
    "uncertain", "uncertainly"
}

FEATURE_COLS_BASE = [
    "polarity",
    "subjectivity",
    "question_count",
    "exclamation_count",
    "word_count",
    "char_count",
    "strong_modality_count",
    "weak_modality_count",
    "strong_modality_density",
    "weak_modality_density",
    "modality_balance",
    "confidence"
]


# =========================================================
# HELPERS
# =========================================================
def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def tokenize(text: str):
    return re.findall(r"[a-zA-Z']+", str(text).lower())


def save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_text(path: Path, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def normalize_count(count, word_count):
    return 0.0 if word_count == 0 else count / word_count


def safe_mean(series):
    series = pd.to_numeric(series, errors="coerce")
    if series.dropna().empty:
        return np.nan
    return series.mean()


def safe_std(series):
    series = pd.to_numeric(series, errors="coerce")
    if series.dropna().empty:
        return np.nan
    return series.std()


def compute_slope(x, y):
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    mask = ~(x.isna() | y.isna())
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return np.nan
    return np.polyfit(x, y, 1)[0]


def assign_phase(turn_position):
    if pd.isna(turn_position):
        return "unknown"
    if turn_position <= 0.33:
        return "early"
    elif turn_position <= 0.66:
        return "middle"
    return "late"


def plot_line(df, x_col, y_cols, title, outpath, xlabel="Turn Index", ylabel="Value"):
    if df.empty or not y_cols:
        return

    plt.figure(figsize=(10, 5))
    for col in y_cols:
        if col in df.columns:
            plt.plot(df[x_col], df[col], marker="o", label=col)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_bar(series, title, outpath, xlabel="", ylabel="Count"):
    if series.empty:
        return
    plt.figure(figsize=(10, 5))
    series.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_boxplot(df, group_col, value_col, title, outpath):
    if df.empty or group_col not in df.columns or value_col not in df.columns:
        return
    plot_df = df[[group_col, value_col]].dropna()
    if plot_df.empty:
        return

    grouped = [vals[value_col].values for _, vals in plot_df.groupby(group_col)]
    labels = [str(k) for k, _ in plot_df.groupby(group_col)]

    if len(grouped) < 2:
        return

    plt.figure(figsize=(8, 5))
    plt.boxplot(grouped, labels=labels)
    plt.title(title)
    plt.ylabel(value_col)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


def plot_grouped_bar(df, category_col, value_col, hue_col, title, outpath):
    if df.empty:
        return

    pivot = df.pivot(index=category_col, columns=hue_col, values=value_col)
    if pivot.empty:
        return

    pivot.plot(kind="bar", figsize=(10, 5))
    plt.title(title)
    plt.ylabel(value_col)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close()


# =========================================================
# LOAD NRC
# =========================================================
def load_nrc(path: Path):
    lex = defaultdict(set)

    if not path.exists():
        raise FileNotFoundError(f"NRC lexicon not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = re.split(r"\s+", line)
            if len(parts) != 3:
                continue

            word, emotion, assoc = parts
            if assoc == "1":
                lex[word.lower()].add(emotion.lower())

    return lex


print("Loading NRC emotion lexicon...")
NRC_LEXICON = load_nrc(NRC_PATH)
print(f"NRC loaded with {len(NRC_LEXICON)} word entries.\n")


# =========================================================
# LOAD BERT EMOTION MODEL
# =========================================================
print("Loading BERT emotion model...")
emotion_classifier = pipeline(
    "text-classification",
    model=EMOTION_MODEL_NAME,
    top_k=None
)
print("BERT emotion model loaded.\n")


# =========================================================
# WINNER INFERENCE
# =========================================================
def infer_winner_from_final_report(final_report):
    if not final_report:
        return {
            "winner_inferred": None,
            "winner_role": None,
            "winner_source": None,
            "winner_confidence": "low",
            "winner_evidence": None
        }

    outcome_summary = str(final_report.get("outcome_summary", "")).strip()
    text = outcome_summary.lower()

    winner = None
    confidence = "low"
    evidence = outcome_summary

    if re.search(r"\bsa successfully defended\b", text):
        winner = "SA"
        confidence = "high"
    elif re.search(r"\bca successfully defended\b", text):
        winner = "CA"
        confidence = "high"
    elif re.search(r"\bsa won\b|\bscientific advocate won\b", text):
        winner = "SA"
        confidence = "high"
    elif re.search(r"\bca won\b|\bconspiracy advocate won\b", text):
        winner = "CA"
        confidence = "high"
    elif re.search(r"\bstrengthening sa'?s position\b", text):
        winner = "SA"
        confidence = "medium"
    elif re.search(r"\bstrengthening ca'?s position\b", text):
        winner = "CA"
        confidence = "medium"
    elif "shift in the debate towards their stance" in text:
        if "sa" in text:
            winner = "SA"
            confidence = "medium"
        elif "ca" in text:
            winner = "CA"
            confidence = "medium"

    role_map = {
        "CA": "proponent",
        "SA": "opponent"
    }

    return {
        "winner_inferred": winner,
        "winner_role": role_map.get(winner),
        "winner_source": "outcome_summary" if outcome_summary else None,
        "winner_confidence": confidence,
        "winner_evidence": evidence if winner else None
    }


def infer_winner_from_stance_trajectory(final_report):
    traj = final_report.get("stance_trajectory", {})
    ca = traj.get("CA", [])
    sa = traj.get("SA", [])

    if len(ca) >= 2 and len(sa) >= 2:
        ca_change = ca[-1] - ca[0]
        sa_change = sa[-1] - sa[0]

        if sa_change > ca_change and sa[-1] > ca[-1]:
            return {
                "winner_inferred": "SA",
                "winner_role": "opponent",
                "winner_source": "stance_trajectory",
                "winner_confidence": "medium",
                "winner_evidence": {
                    "CA_start": ca[0], "CA_end": ca[-1],
                    "SA_start": sa[0], "SA_end": sa[-1]
                }
            }

        if ca_change > sa_change and ca[-1] > sa[-1]:
            return {
                "winner_inferred": "CA",
                "winner_role": "proponent",
                "winner_source": "stance_trajectory",
                "winner_confidence": "medium",
                "winner_evidence": {
                    "CA_start": ca[0], "CA_end": ca[-1],
                    "SA_start": sa[0], "SA_end": sa[-1]
                }
            }

    return {
        "winner_inferred": None,
        "winner_role": None,
        "winner_source": None,
        "winner_confidence": "low",
        "winner_evidence": None
    }


# =========================================================
# PER-UTTERANCE ANALYSIS
# =========================================================
def analyze_utterance(text: str):
    text = str(text)
    tokens = tokenize(text)
    word_count = len(tokens)
    blob = TextBlob(text)

    scores = {
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity,
        "question_count": text.count("?"),
        "exclamation_count": text.count("!"),
        "word_count": word_count,
        "char_count": len(text),
    }

    strong_count = sum(1 for t in tokens if t in STRONG_MODALITY_WORDS)
    weak_count = sum(1 for t in tokens if t in WEAK_MODALITY_WORDS)

    scores["strong_modality_count"] = strong_count
    scores["weak_modality_count"] = weak_count
    scores["strong_modality_density"] = normalize_count(strong_count, word_count)
    scores["weak_modality_density"] = normalize_count(weak_count, word_count)
    scores["modality_balance"] = strong_count - weak_count

    emotion_counts = Counter()
    for t in tokens:
        if t in NRC_LEXICON:
            for emotion in NRC_LEXICON[t]:
                emotion_counts[emotion] += 1

    for emotion in NRC_TRUE_EMOTIONS:
        count = emotion_counts.get(emotion, 0)
        scores[f"nrc_{emotion}_count"] = count
        scores[f"nrc_{emotion}_density"] = normalize_count(count, word_count)

    try:
        emotion_results = emotion_classifier(text)[0]
        for item in emotion_results:
            label = item["label"].lower().replace(" ", "_")
            scores[f"emotion_{label}"] = item["score"]
    except Exception as e:
        scores["emotion_model_error"] = str(e)

    return pd.Series(scores)


# =========================================================
# FEATURE BUILDERS
# =========================================================
def get_top_emotion_label(df, cols, prefix_to_strip):
    if not cols:
        return None
    means = df[cols].mean(numeric_only=True)
    if means.empty:
        return None
    top_col = means.sort_values(ascending=False).index[0]
    return top_col.replace(prefix_to_strip, "")


def build_speaker_features(df_speaker, debate_id, claim, speaker_role, winner_role, emotion_cols, nrc_density_cols):
    row = {
        "debate_id": debate_id,
        "claim": claim,
        "speaker_role": speaker_role,
        "is_winner": int(speaker_role == winner_role) if winner_role else np.nan,
        "n_turns": len(df_speaker),
    }

    for col in FEATURE_COLS_BASE:
        if col in df_speaker.columns:
            row[f"{col}_mean"] = safe_mean(df_speaker[col])
            row[f"{col}_std"] = safe_std(df_speaker[col])
            row[f"{col}_slope"] = compute_slope(df_speaker["turn_index"], df_speaker[col])

    row["turn_position_mean"] = safe_mean(df_speaker["turn_position"])

    for phase in ["early", "middle", "late"]:
        phase_df = df_speaker[df_speaker["phase"] == phase]
        for col in ["confidence", "strong_modality_density", "weak_modality_density", "modality_balance", "polarity", "subjectivity"]:
            if col in df_speaker.columns:
                row[f"{phase}_{col}_mean"] = safe_mean(phase_df[col])

    for col in ["confidence", "strong_modality_density", "weak_modality_density", "modality_balance", "polarity", "subjectivity"]:
        early_val = row.get(f"early_{col}_mean")
        late_val = row.get(f"late_{col}_mean")
        if pd.notna(early_val) and pd.notna(late_val):
            row[f"late_minus_early_{col}"] = late_val - early_val
        else:
            row[f"late_minus_early_{col}"] = np.nan

    if emotion_cols:
        for col in emotion_cols:
            row[f"{col}_mean"] = safe_mean(df_speaker[col])
        row["top_bert_emotion"] = get_top_emotion_label(df_speaker, emotion_cols, "emotion_")
        row["dominant_bert_mode"] = (
            df_speaker["dominant_bert_emotion"].mode().iloc[0]
            if "dominant_bert_emotion" in df_speaker.columns and not df_speaker["dominant_bert_emotion"].mode().empty
            else None
        )

    if nrc_density_cols:
        for col in nrc_density_cols:
            row[f"{col}_mean"] = safe_mean(df_speaker[col])
        row["top_nrc_emotion"] = get_top_emotion_label(df_speaker, nrc_density_cols, "nrc_")
        if row["top_nrc_emotion"]:
            row["top_nrc_emotion"] = row["top_nrc_emotion"].replace("_density", "")
        row["dominant_nrc_mode"] = (
            df_speaker["dominant_nrc_emotion"].mode().iloc[0]
            if "dominant_nrc_emotion" in df_speaker.columns and not df_speaker["dominant_nrc_emotion"].mode().empty
            else None
        )

    return row


def build_debate_level_features(df_enriched, debate_id, claim, winner_info):
    winner_role = winner_info.get("winner_role")
    roles = ["proponent", "opponent"]

    row = {
        "debate_id": debate_id,
        "claim": claim,
        "winner_role": winner_role,
        "winner_source": winner_info.get("winner_source"),
        "winner_confidence_label": winner_info.get("winner_confidence"),
        "n_total_turns": len(df_enriched)
    }

    speaker_dfs = {role: df_enriched[df_enriched["speaker_role"] == role].copy() for role in roles}

    metrics = ["confidence", "strong_modality_density", "weak_modality_density", "modality_balance", "polarity", "subjectivity", "word_count"]

    for role in roles:
        sdf = speaker_dfs[role]
        row[f"{role}_turns"] = len(sdf)
        for metric in metrics:
            if metric in sdf.columns:
                row[f"{role}_{metric}_mean"] = safe_mean(sdf[metric])

        late = sdf[sdf["phase"] == "late"]
        for metric in ["confidence", "strong_modality_density", "weak_modality_density", "modality_balance"]:
            row[f"{role}_late_{metric}_mean"] = safe_mean(late[metric]) if metric in sdf.columns else np.nan

    if winner_role in roles:
        loser_role = "opponent" if winner_role == "proponent" else "proponent"
        row["loser_role"] = loser_role

        for metric in metrics:
            w = row.get(f"{winner_role}_{metric}_mean")
            l = row.get(f"{loser_role}_{metric}_mean")
            row[f"winner_minus_loser_{metric}_mean"] = (w - l) if pd.notna(w) and pd.notna(l) else np.nan

        for metric in ["confidence", "strong_modality_density", "weak_modality_density", "modality_balance"]:
            w = row.get(f"{winner_role}_late_{metric}_mean")
            l = row.get(f"{loser_role}_late_{metric}_mean")
            row[f"winner_minus_loser_late_{metric}_mean"] = (w - l) if pd.notna(w) and pd.notna(l) else np.nan
    else:
        row["loser_role"] = None

    return row


# =========================================================
# SINGLE RUN PROCESSING
# =========================================================
def process_single_run(run_dir: Path):
    suffix = run_dir.name.replace("run_", "")
    analysis_dir = OUTPUT_ANALYSIS_DIR / f"analysis_{suffix}"

    if analysis_dir.exists() and not OVERWRITE_EXISTING:
        print(f"Skipping {run_dir.name}: analysis already exists.")
        return None

    ensure_dir(analysis_dir)
    plots_dir = analysis_dir / "plots"
    ensure_dir(plots_dir)

    debate_log_path = run_dir / "debate_log.csv"
    final_report_path = run_dir / "final_report.json"

    if not debate_log_path.exists():
        print(f"Skipping {run_dir.name}: debate_log.csv not found.")
        return None

    print(f"Processing {run_dir.name}...")
    df = pd.read_csv(debate_log_path)
    original_rows = len(df)

    if "speaker_role" in df.columns:
        df_agents = df[df["speaker_role"].isin(["proponent", "opponent"])].copy()
        if not df_agents.empty:
            df = df_agents

    df = df.reset_index(drop=True)
    df["turn_index"] = range(1, len(df) + 1)

    if "utterance" not in df.columns:
        raise ValueError(f"{debate_log_path} does not contain an 'utterance' column.")

    if "speaker_role" not in df.columns:
        df["speaker_role"] = "unknown"

    if "round" not in df.columns:
        df["round"] = None

    if "confidence" not in df.columns:
        df["confidence"] = np.nan
    else:
        df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    debate_id = df["debate_id"].iloc[0] if "debate_id" in df.columns and not df["debate_id"].isna().all() else run_dir.name
    claim = df["claim"].iloc[0] if "claim" in df.columns and not df["claim"].isna().all() else None

    analysis = df["utterance"].apply(analyze_utterance)
    df_enriched = pd.concat([df, analysis], axis=1)

    numeric_cols_all = df_enriched.select_dtypes(include="number").columns.tolist()
    df_enriched[numeric_cols_all] = df_enriched[numeric_cols_all].fillna(0)

    total_turns = len(df_enriched)
    if total_turns > 1:
        df_enriched["turn_position"] = (df_enriched["turn_index"] - 1) / (total_turns - 1)
    else:
        df_enriched["turn_position"] = 0.0

    df_enriched["phase"] = df_enriched["turn_position"].apply(assign_phase)

    nrc_density_cols = [
        c for c in df_enriched.columns
        if c.startswith("nrc_")
        and c.endswith("_density")
        and c.replace("nrc_", "").replace("_density", "") in NRC_TRUE_EMOTIONS
    ]

    if nrc_density_cols:
        df_enriched["dominant_nrc_emotion"] = (
            df_enriched[nrc_density_cols]
            .idxmax(axis=1)
            .str.replace("nrc_", "", regex=False)
            .str.replace("_density", "", regex=False)
        )
    else:
        df_enriched["dominant_nrc_emotion"] = "unknown"

    emotion_cols = [c for c in df_enriched.columns if c.startswith("emotion_") and c != "emotion_model_error"]

    if emotion_cols:
        df_enriched[emotion_cols] = df_enriched[emotion_cols].fillna(0.0)
        df_enriched["dominant_bert_emotion"] = (
            df_enriched[emotion_cols]
            .idxmax(axis=1)
            .str.replace("emotion_", "", regex=False)
        )
    else:
        df_enriched["dominant_bert_emotion"] = "unknown"

    numeric_cols = df_enriched.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in ["round", "turn_index"]]

    speaker_summary = df_enriched.groupby("speaker_role")[numeric_cols].mean(numeric_only=True)
    phase_summary = (
        df_enriched.groupby(["speaker_role", "phase"])[numeric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )

    phase_summary.insert(0, "debate_id", debate_id)
    phase_summary.insert(1, "claim", claim)

    dominant_bert_emotion_by_speaker = (
        df_enriched.groupby("speaker_role")["dominant_bert_emotion"]
        .value_counts()
        .rename("count")
        .reset_index()
    )

    dominant_nrc_emotion_by_speaker = (
        df_enriched.groupby("speaker_role")["dominant_nrc_emotion"]
        .value_counts()
        .rename("count")
        .reset_index()
    )

    final_report = None
    if final_report_path.exists():
        with open(final_report_path, "r", encoding="utf-8") as f:
            final_report = json.load(f)

    winner_info = infer_winner_from_final_report(final_report)
    if winner_info["winner_inferred"] is None and final_report:
        winner_info = infer_winner_from_stance_trajectory(final_report)

    winner_role = winner_info.get("winner_role")

    analysis_metadata = {
        "input_run_folder": str(run_dir),
        "output_analysis_folder": str(analysis_dir),
        "debate_id": debate_id,
        "claim": claim,
        "original_row_count": int(original_rows),
        "analyzed_row_count": int(len(df_enriched)),
        "speaker_roles_found": sorted(df_enriched["speaker_role"].dropna().astype(str).unique().tolist()),
        "nrc_path": str(NRC_PATH),
        "nrc_word_entries": len(NRC_LEXICON),
        "bert_model_name": EMOTION_MODEL_NAME,
        "winner_info": winner_info,
        "final_report_present": final_report is not None
    }

    speaker_feature_rows = []
    for speaker in ["proponent", "opponent"]:
        sub = df_enriched[df_enriched["speaker_role"] == speaker].copy()
        if sub.empty:
            continue
        speaker_feature_rows.append(
            build_speaker_features(
                sub, debate_id, claim, speaker, winner_role, emotion_cols, nrc_density_cols
            )
        )

    debate_feature_row = build_debate_level_features(df_enriched, debate_id, claim, winner_info)

    df_enriched.to_csv(analysis_dir / "debate_log_with_modality_emotion.csv", index=False)
    speaker_summary.to_csv(analysis_dir / "speaker_summary.csv")
    phase_summary.to_csv(analysis_dir / "phase_summary.csv", index=False)
    dominant_bert_emotion_by_speaker.to_csv(analysis_dir / "dominant_bert_emotion_by_speaker.csv", index=False)
    dominant_nrc_emotion_by_speaker.to_csv(analysis_dir / "dominant_nrc_emotion_by_speaker.csv", index=False)
    pd.DataFrame(speaker_feature_rows).to_csv(analysis_dir / "speaker_debate_features.csv", index=False)
    pd.DataFrame([debate_feature_row]).to_csv(analysis_dir / "debate_level_features.csv", index=False)

    save_json(analysis_dir / "winner_summary.json", winner_info)
    save_json(analysis_dir / "analysis_metadata.json", analysis_metadata)

    report_lines = []
    report_lines.append("=== ANALYSIS METADATA ===")
    report_lines.append(json.dumps(analysis_metadata, indent=2))
    report_lines.append("\n=== WINNER INFO ===")
    report_lines.append(json.dumps(winner_info, indent=2))
    report_lines.append("\n=== SPEAKER SUMMARY ===")
    report_lines.append(speaker_summary.to_string())
    report_lines.append("\n=== PHASE SUMMARY ===")
    report_lines.append(phase_summary.to_string(index=False))
    report_lines.append("\n=== DEBATE LEVEL FEATURES ===")
    report_lines.append(pd.DataFrame([debate_feature_row]).to_string(index=False))
    save_text(analysis_dir / "analysis_output.txt", "\n".join(report_lines))

    for speaker in df_enriched["speaker_role"].dropna().unique():
        sub = df_enriched[df_enriched["speaker_role"] == speaker].copy()
        safe_speaker = str(speaker).replace(" ", "_").lower()

        modality_cols = [c for c in [
            "strong_modality_density",
            "weak_modality_density",
            "modality_balance",
            "confidence"
        ] if c in sub.columns]

        if modality_cols:
            plot_line(
                sub,
                x_col="turn_index",
                y_cols=modality_cols,
                title=f"{speaker} - Modality / Confidence Over Time",
                outpath=plots_dir / f"{safe_speaker}_modality_confidence_over_time.png"
            )

        top_bert_emotions = []
        if emotion_cols:
            top_bert_emotions = (
                sub[emotion_cols].mean().sort_values(ascending=False).head(5).index.tolist()
            )
        if top_bert_emotions:
            plot_line(
                sub,
                x_col="turn_index",
                y_cols=top_bert_emotions,
                title=f"{speaker} - Top BERT Emotions Over Time",
                outpath=plots_dir / f"{safe_speaker}_top_bert_emotions_over_time.png"
            )

        phase_counts = sub["phase"].value_counts().sort_index()
        if not phase_counts.empty:
            plot_bar(
                phase_counts,
                title=f"{speaker} - Turn Distribution by Phase",
                outpath=plots_dir / f"{safe_speaker}_phase_counts.png",
                xlabel="Phase",
                ylabel="Turns"
            )

    for metric in ["strong_modality_density", "weak_modality_density", "modality_balance", "confidence", "polarity", "subjectivity"]:
        if metric in df_enriched.columns:
            plt.figure(figsize=(10, 5))
            for speaker in df_enriched["speaker_role"].dropna().unique():
                sub = df_enriched[df_enriched["speaker_role"] == speaker]
                plt.plot(sub["turn_index"], sub[metric], marker="o", label=str(speaker))
            plt.title(f"Speaker Comparison - {metric}")
            plt.xlabel("Turn Index")
            plt.ylabel(metric)
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / f"comparison_{metric}.png", dpi=300, bbox_inches="tight")
            plt.close()

    print(f"Finished {run_dir.name} -> {analysis_dir}")

    return {
        "speaker_features": speaker_feature_rows,
        "debate_features": debate_feature_row,
        "phase_summary": phase_summary.copy(),
        "turn_data": df_enriched.copy()
    }


# =========================================================
# CROSS-DEBATE AGGREGATION
# =========================================================
def aggregate_all_results(results):
    speaker_rows = []
    debate_rows = []
    phase_rows = []
    turn_rows = []

    for res in results:
        if not res:
            continue
        speaker_rows.extend(res["speaker_features"])
        debate_rows.append(res["debate_features"])
        phase_rows.append(res["phase_summary"])
        turn_rows.append(res["turn_data"])

    if not speaker_rows:
        print("No results available for aggregation.")
        return

    aggregate_dir = OUTPUT_ANALYSIS_DIR / "aggregate_reports"
    ensure_dir(aggregate_dir)
    plots_dir = aggregate_dir / "plots"
    ensure_dir(plots_dir)

    speaker_df = pd.DataFrame(speaker_rows)
    debate_df = pd.DataFrame(debate_rows)
    phase_df = pd.concat(phase_rows, ignore_index=True) if phase_rows else pd.DataFrame()
    turn_df = pd.concat(turn_rows, ignore_index=True) if turn_rows else pd.DataFrame()

    speaker_df.to_csv(aggregate_dir / "speaker_debate_features.csv", index=False)
    debate_df.to_csv(aggregate_dir / "debate_level_features.csv", index=False)
    phase_df.to_csv(aggregate_dir / "phase_summary_all_debates.csv", index=False)
    turn_df.to_csv(aggregate_dir / "turn_level_all_debates.csv", index=False)

    # Winner vs loser summary
    numeric_cols = speaker_df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "is_winner"]

    winner_vs_loser_rows = []
    for col in numeric_cols:
        winners = speaker_df.loc[speaker_df["is_winner"] == 1, col]
        losers = speaker_df.loc[speaker_df["is_winner"] == 0, col]
        winner_vs_loser_rows.append({
            "feature": col,
            "winner_mean": safe_mean(winners),
            "loser_mean": safe_mean(losers),
            "winner_minus_loser": safe_mean(winners) - safe_mean(losers)
            if pd.notna(safe_mean(winners)) and pd.notna(safe_mean(losers)) else np.nan
        })

    winner_vs_loser_df = pd.DataFrame(winner_vs_loser_rows).sort_values(
        "winner_minus_loser", ascending=False
    )
    winner_vs_loser_df.to_csv(aggregate_dir / "winner_vs_loser_summary.csv", index=False)

    # Phase comparison
    if not phase_df.empty:
        phase_metrics = ["confidence", "strong_modality_density", "weak_modality_density", "modality_balance", "polarity", "subjectivity"]
        merged_phase = phase_df.merge(
            speaker_df[["debate_id", "speaker_role", "is_winner"]],
            on=["debate_id", "speaker_role"],
            how="left"
        )

        phase_compare_rows = []
        for phase in ["early", "middle", "late"]:
            for metric in phase_metrics:
                if metric in merged_phase.columns:
                    winners = merged_phase[(merged_phase["phase"] == phase) & (merged_phase["is_winner"] == 1)][metric]
                    losers = merged_phase[(merged_phase["phase"] == phase) & (merged_phase["is_winner"] == 0)][metric]
                    phase_compare_rows.append({
                        "phase": phase,
                        "metric": metric,
                        "winner_mean": safe_mean(winners),
                        "loser_mean": safe_mean(losers),
                        "winner_minus_loser": safe_mean(winners) - safe_mean(losers)
                        if pd.notna(safe_mean(winners)) and pd.notna(safe_mean(losers)) else np.nan
                    })

        phase_compare_df = pd.DataFrame(phase_compare_rows)
        phase_compare_df.to_csv(aggregate_dir / "phase_winner_vs_loser_summary.csv", index=False)

        for metric in ["confidence", "strong_modality_density", "weak_modality_density", "modality_balance"]:
            metric_df = phase_compare_df[phase_compare_df["metric"] == metric]
            if not metric_df.empty:
                long_df = pd.concat([
                    metric_df[["phase", "winner_mean"]].rename(columns={"winner_mean": "value"}).assign(group="winner"),
                    metric_df[["phase", "loser_mean"]].rename(columns={"loser_mean": "value"}).assign(group="loser")
                ], ignore_index=True)

                plot_grouped_bar(
                    long_df,
                    category_col="phase",
                    value_col="value",
                    hue_col="group",
                    title=f"Winners vs Losers by Phase - {metric}",
                    outpath=plots_dir / f"phase_winner_vs_loser_{metric}.png"
                )

    # Boxplots
    for metric in [
        "confidence_mean",
        "strong_modality_density_mean",
        "weak_modality_density_mean",
        "modality_balance_mean",
        "late_minus_early_confidence",
        "late_minus_early_strong_modality_density",
        "late_minus_early_weak_modality_density"
    ]:
        if metric in speaker_df.columns:
            plot_boxplot(
                speaker_df,
                group_col="is_winner",
                value_col=metric,
                title=f"Winners vs Losers - {metric}",
                outpath=plots_dir / f"boxplot_{metric}.png"
            )

    # Trajectory summary using turn_position bins
    if not turn_df.empty:
        turn_df["turn_bin"] = pd.cut(
            turn_df["turn_position"],
            bins=np.linspace(0, 1, 6),
            include_lowest=True,
            labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"]
        )

        turn_df = turn_df.merge(
            speaker_df[["debate_id", "speaker_role", "is_winner"]],
            on=["debate_id", "speaker_role"],
            how="left"
        )

        for metric in ["confidence", "strong_modality_density", "weak_modality_density", "modality_balance"]:
            if metric not in turn_df.columns:
                continue

            traj = (
                turn_df.groupby(["turn_bin", "is_winner"])[metric]
                .mean()
                .reset_index()
            )

            if traj.empty:
                continue

            pivot = traj.pivot(index="turn_bin", columns="is_winner", values=metric)
            if pivot.empty:
                continue

            plt.figure(figsize=(10, 5))
            if 1 in pivot.columns:
                plt.plot(pivot.index.astype(str), pivot[1], marker="o", label="winner")
            if 0 in pivot.columns:
                plt.plot(pivot.index.astype(str), pivot[0], marker="o", label="loser")
            plt.title(f"Average Trajectory - {metric}")
            plt.xlabel("Debate Progress")
            plt.ylabel(metric)
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / f"trajectory_{metric}.png", dpi=300, bbox_inches="tight")
            plt.close()

    # Simple text report
    top_positive = winner_vs_loser_df.head(10)
    top_negative = winner_vs_loser_df.tail(10)

    lines = []
    lines.append("=== WINNER VS LOSER SUMMARY ===")
    lines.append("Top features where winners are higher:")
    lines.append(top_positive.to_string(index=False))
    lines.append("\nTop features where losers are higher:")
    lines.append(top_negative.to_string(index=False))

    if "winner_minus_loser_late_confidence_mean" in debate_df.columns:
        lines.append("\n=== DEBATE-LEVEL AVERAGES ===")
        lines.append(debate_df.mean(numeric_only=True).to_string())

    save_text(aggregate_dir / "aggregate_analysis_report.txt", "\n".join(lines))

    print(f"\nAggregate reports saved to: {aggregate_dir}")


# =========================================================
# MAIN
# =========================================================
def main():
    if not INPUT_RUNS_DIR.exists():
        raise FileNotFoundError(f"Could not find input folder: {INPUT_RUNS_DIR}")

    ensure_dir(OUTPUT_ANALYSIS_DIR)

    run_dirs = sorted(
        [p for p in INPUT_RUNS_DIR.iterdir() if p.is_dir() and p.name.startswith("run_")]
    )

    if not run_dirs:
        print("No run_* folders found.")
        return

    print(f"Found {len(run_dirs)} run folders.")
    print("Processing all runs:")
    for run_dir in run_dirs:
        print(f" - {run_dir.name}")
    print()

    results = []
    for run_dir in run_dirs:
        try:
            res = process_single_run(run_dir)
            if res is not None:
                results.append(res)
        except Exception as e:
            print(f"ERROR while processing {run_dir.name}: {e}")

    aggregate_all_results(results)
    print("\nBatch modality + emotion + winner analysis complete.")


if __name__ == "__main__":
    main()