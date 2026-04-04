import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from textblob import TextBlob
from transformers import pipeline

# =========================================================
# CONFIG
# =========================================================
INPUT_RUNS_DIR = Path("old_artifacts")
OUTPUT_ANALYSIS_DIR = Path("analysis_outputs")
MAX_RUNS = 5

EMOTION_MODEL_NAME = "bhadresh-savani/bert-base-uncased-emotion"
OVERWRITE_EXISTING = True

UNCERTAINTY_WORDS = {
    "may", "might", "could", "perhaps", "possibly", "suggests",
    "likely", "potential", "concern", "concerns", "reasonable",
    "appears", "seems", "arguably", "maybe", "unclear", "uncertain"
}

STRONG_MODALITY = {
    "must", "clearly", "definitely", "proves", "demonstrates",
    "cannot", "never", "always", "undeniable", "certainly",
    "obviously", "plainly", "shows", "confirms"
}

WEAK_MODALITY = {
    "may", "might", "could", "suggest", "suggests", "appears",
    "seems", "possibly", "potential", "perhaps", "arguably"
}


# =========================================================
# MODEL SETUP
# =========================================================
print("Loading emotion model...")
emotion_classifier = pipeline(
    "text-classification",
    model=EMOTION_MODEL_NAME,
    top_k=None
)
print("Emotion model loaded.\n")


# =========================================================
# HELPERS
# =========================================================
def tokenize(text: str):
    return re.findall(r"[a-zA-Z']+", str(text).lower())


def safe_read_json(path: Path):
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def analyze_utterance(text: str):
    text = str(text)
    tokens = tokenize(text)
    blob = TextBlob(text)

    scores = {
        "polarity": blob.sentiment.polarity,
        "subjectivity": blob.sentiment.subjectivity,
        "uncertainty_score": sum(t in UNCERTAINTY_WORDS for t in tokens),
        "strong_modality_score": sum(t in STRONG_MODALITY for t in tokens),
        "weak_modality_score": sum(t in WEAK_MODALITY for t in tokens),
        "question_count": text.count("?"),
        "exclamation_count": text.count("!"),
        "word_count": len(tokens),
        "char_count": len(text),
    }

    try:
        emotion_results = emotion_classifier(text)[0]
        for item in emotion_results:
            label = item["label"].lower().replace(" ", "_")
            scores[f"emotion_{label}"] = item["score"]
    except Exception as e:
        scores["emotion_model_error"] = str(e)

    return pd.Series(scores)


def save_text_report(path: Path, sections):
    with open(path, "w", encoding="utf-8") as f:
        for title, content in sections:
            f.write(title + "\n")
            f.write(content + "\n\n")


def plot_line(df, x_col, y_cols, title, outpath, xlabel="Turn Index", ylabel="Score"):
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


def process_single_run(run_dir: Path):
    suffix = run_dir.name.replace("run_", "")
    analysis_dir = OUTPUT_ANALYSIS_DIR / f"analysis_{suffix}"

    if analysis_dir.exists() and not OVERWRITE_EXISTING:
        print(f"Skipping {run_dir.name}: analysis already exists.")
        return

    ensure_dir(analysis_dir)
    plots_dir = analysis_dir / "plots"
    ensure_dir(plots_dir)

    debate_log_path = run_dir / "debate_log.csv"
    transcript_path = run_dir / "transcript.md"
    memory_path = run_dir / "memory.json"
    final_report_path = run_dir / "final_report.json"
    metrics_path = run_dir / "metrics.csv"
    experiment_metadata_path = run_dir / "experiment_metadata.json"
    run_config_path = run_dir / "run_config.json"

    final_report = safe_read_json(final_report_path)
    winner_info = infer_winner_from_final_report(final_report)

    if not debate_log_path.exists():
        print(f"Skipping {run_dir.name}: debate_log.csv not found.")
        return

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
        df["confidence"] = None

    analysis = df["utterance"].apply(analyze_utterance)
    df_enriched = pd.concat([df, analysis], axis=1)

    emotion_cols = [c for c in df_enriched.columns if c.startswith("emotion_") and c != "emotion_model_error"]
    if emotion_cols:
        df_enriched[emotion_cols] = df_enriched[emotion_cols].fillna(0.0)
        df_enriched["dominant_emotion"] = (
            df_enriched[emotion_cols]
            .idxmax(axis=1)
            .str.replace("emotion_", "", regex=False)
        )
    else:
        df_enriched["dominant_emotion"] = "unknown"

    numeric_cols = df_enriched.select_dtypes(include="number").columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in ["round"]]

    speaker_summary = pd.DataFrame()
    if "speaker_role" in df_enriched.columns:
        speaker_summary = df_enriched.groupby("speaker_role")[numeric_cols].mean(numeric_only=True)

    round_summary = pd.DataFrame()
    if "round" in df_enriched.columns and "speaker_role" in df_enriched.columns:
        round_summary = (
            df_enriched.groupby(["round", "speaker_role"])[numeric_cols]
            .mean(numeric_only=True)
            .reset_index()
        )

    dominant_emotion_by_speaker = pd.DataFrame()
    if "speaker_role" in df_enriched.columns:
        dominant_emotion_by_speaker = (
            df_enriched.groupby("speaker_role")["dominant_emotion"]
            .value_counts()
            .rename("count")
            .reset_index()
        )

    run_summary = {
        "input_run_folder": str(run_dir),
        "output_analysis_folder": str(analysis_dir),
        "original_row_count": int(original_rows),
        "analyzed_row_count": int(len(df_enriched)),
        "speaker_roles_found": sorted(df_enriched["speaker_role"].dropna().astype(str).unique().tolist()),
        "emotion_columns": emotion_cols,
        "winner_info": winner_info,
        "files_found": {
            "debate_log.csv": debate_log_path.exists(),
            "transcript.md": transcript_path.exists(),
            "memory.json": memory_path.exists(),
            "final_report.json": final_report_path.exists(),
            "metrics.csv": metrics_path.exists(),
            "experiment_metadata.json": experiment_metadata_path.exists(),
            "run_config.json": run_config_path.exists(),
        }
    }

    write_json(analysis_dir / "analysis_metadata.json", run_summary)
    write_json(analysis_dir / "winner_summary.json", winner_info)

    # Save copies of useful JSON inputs into your branch's analysis folder
    for src in [experiment_metadata_path, run_config_path, final_report_path, memory_path]:
        if src.exists():
            data = safe_read_json(src)
            if data is not None:
                write_json(analysis_dir / f"copy_{src.name}", data)

    # Save enriched outputs
    df_enriched.to_csv(analysis_dir / "debate_log_with_detailed_emotions.csv", index=False)

    if not speaker_summary.empty:
        speaker_summary.to_csv(analysis_dir / "speaker_summary.csv")

    if not round_summary.empty:
        round_summary.to_csv(analysis_dir / "round_summary.csv", index=False)

    if not dominant_emotion_by_speaker.empty:
        dominant_emotion_by_speaker.to_csv(analysis_dir / "dominant_emotion_by_speaker.csv", index=False)

    sections = [
        ("=== RUN SUMMARY ===", json.dumps(run_summary, indent=2)),
        ("=== SPEAKER SUMMARY ===", speaker_summary.to_string() if not speaker_summary.empty else "No speaker summary available."),
        ("=== ROUND SUMMARY ===", round_summary.to_string(index=False) if not round_summary.empty else "No round summary available."),
        ("=== DOMINANT EMOTION BY SPEAKER ===", dominant_emotion_by_speaker.to_string(index=False) if not dominant_emotion_by_speaker.empty else "No dominant emotion data available.")
    ]
    save_text_report(analysis_dir / "analysis_output.txt", sections)

    for speaker in df_enriched["speaker_role"].dropna().unique():
        sub = df_enriched[df_enriched["speaker_role"] == speaker].copy()
        safe_speaker = str(speaker).replace(" ", "_").lower()

        cols = [c for c in ["confidence", "polarity", "subjectivity"] if c in sub.columns]
        if cols:
            plot_line(
                sub,
                x_col="turn_index",
                y_cols=cols,
                title=f"{speaker} - Confidence / Polarity / Subjectivity",
                outpath=plots_dir / f"{safe_speaker}_confidence_polarity_subjectivity.png"
            )

        speaker_emotion_cols = [c for c in emotion_cols if c in sub.columns]
        if speaker_emotion_cols:
            top_emotions = (
                sub[speaker_emotion_cols]
                .mean()
                .sort_values(ascending=False)
                .head(5)
                .index
                .tolist()
            )
            plot_line(
                sub,
                x_col="turn_index",
                y_cols=top_emotions,
                title=f"{speaker} - Top Emotions Over Time",
                outpath=plots_dir / f"{safe_speaker}_top_emotions_over_time.png"
            )

        rhetoric_cols = [
            c for c in [
                "uncertainty_score",
                "strong_modality_score",
                "weak_modality_score",
                "question_count",
                "exclamation_count"
            ] if c in sub.columns
        ]
        if rhetoric_cols:
            plot_line(
                sub,
                x_col="turn_index",
                y_cols=rhetoric_cols,
                title=f"{speaker} - Rhetorical Markers Over Time",
                outpath=plots_dir / f"{safe_speaker}_rhetorical_markers.png"
            )

        if "dominant_emotion" in sub.columns:
            emotion_counts = sub["dominant_emotion"].value_counts()
            if not emotion_counts.empty:
                plot_bar(
                    emotion_counts,
                    title=f"{speaker} - Dominant Emotion Counts",
                    outpath=plots_dir / f"{safe_speaker}_dominant_emotion_counts.png",
                    xlabel="Emotion",
                    ylabel="Number of Turns"
                )

    for metric in ["polarity", "subjectivity", "confidence"]:
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

    print(f"Finished {run_dir.name} -> {analysis_dir}\n")


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

    # Strong patterns
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
    elif re.search(r"\bshift in the debate towards their stance\b", text):
        # Try to resolve "their" from earlier sentence
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

    selected_runs = run_dirs[:MAX_RUNS]

    print(f"Found {len(run_dirs)} run folders.")
    print(f"Processing first {len(selected_runs)} runs:\n")
    for run_dir in selected_runs:
        print(f" - {run_dir.name}")
    print()

    for run_dir in selected_runs:
        try:
            process_single_run(run_dir)
        except Exception as e:
            print(f"ERROR while processing {run_dir.name}: {e}\n")

    print("Batch analysis complete.")


if __name__ == "__main__":
    main()
