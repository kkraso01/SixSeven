from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

NRC_EMOTIONS = [
    "anger",
    "anticipation",
    "disgust",
    "fear",
    "joy",
    "sadness",
    "surprise",
    "trust",
]

EMOTION_COLORS = {
    "emotion_anger": "#d62728",
    "emotion_anticipation": "#ff7f0e",
    "emotion_disgust": "#2ca02c",
    "emotion_fear": "#9467bd",
    "emotion_joy": "#ffd700",
    "emotion_sadness": "#1f77b4",
    "emotion_surprise": "#17becf",
    "emotion_trust": "#8c564b",
}

GROUP_HATCHES = ["", "//", "xx", "..", "++", "\\\\"]
ACTOR_LINE_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]


ROLE_EXPECTED_STANCE = {
    "proponent": 1.0,
    "opponent": -1.0,
    "moderator": 0.0,
}


STANCE_LABEL_MAP = {
    "pro": 1.0,
    "support": 1.0,
    "oppose": -1.0,
    "opposed": -1.0,
    "con": -1.0,
    "neutral": 0.0,
    "mixed": 0.0,
}


def _normalize_rows(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    values = df[cols].fillna(0.0).astype(float)
    row_sums = values.sum(axis=1).replace(0.0, np.nan)
    return values.div(row_sums, axis=0).fillna(0.0)


def _build_group_series(df: pd.DataFrame, x_col: str, emotion_cols: list[str]) -> dict[str, pd.DataFrame]:
    grouped: dict[str, pd.DataFrame] = {}
    for (model_name, debate_role), sub in df.groupby(["model_name", "debate_role"], dropna=False):
        actor_key = f"{str(model_name)}::{str(debate_role)}"
        work = sub[[x_col] + emotion_cols].copy()
        work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
        work = work.dropna(subset=[x_col])
        if work.empty:
            continue
        work = work.groupby(x_col, as_index=False)[emotion_cols].mean(numeric_only=True).sort_values(x_col)
        if work.empty:
            continue
        grouped[actor_key] = work
    return grouped


def _build_group_metric_series(df: pd.DataFrame, x_col: str, metric_col: str) -> dict[str, pd.DataFrame]:
    grouped: dict[str, pd.DataFrame] = {}
    for (model_name, debate_role), sub in df.groupby(["model_name", "debate_role"], dropna=False):
        actor_key = f"{str(model_name)}::{str(debate_role)}"
        work = sub[[x_col, metric_col]].copy()
        work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
        work[metric_col] = pd.to_numeric(work[metric_col], errors="coerce")
        work = work.dropna(subset=[x_col, metric_col])
        if work.empty:
            continue
        work = work.groupby(x_col, as_index=False)[metric_col].mean(numeric_only=True).sort_values(x_col)
        if work.empty:
            continue
        grouped[actor_key] = work
    return grouped


def _stance_to_numeric(value: object) -> float:
    key = str(value).strip().lower()
    return float(STANCE_LABEL_MAP.get(key, np.nan))


def plot_combined_for_run(run_dir: Path) -> bool:
    features_path = run_dir / "utterance_features.csv"
    if not features_path.exists():
        return False

    df = pd.read_csv(features_path)
    if df.empty or "model_name" not in df.columns or "debate_role" not in df.columns:
        return False

    x_col = "round" if "round" in df.columns and df["round"].notna().any() else "turn_index"
    if x_col not in df.columns:
        return False

    emotion_cols = [f"emotion_{name}" for name in NRC_EMOTIONS if f"emotion_{name}" in df.columns]
    if not emotion_cols:
        return False

    grouped = _build_group_series(df, x_col=x_col, emotion_cols=emotion_cols)
    if not grouped:
        return False

    all_steps = sorted(
        {
            int(step) if float(step).is_integer() else float(step)
            for actor_df in grouped.values()
            for step in actor_df[x_col].dropna().tolist()
        }
    )
    if not all_steps:
        return False

    actor_keys = sorted(grouped.keys())
    n_groups = len(actor_keys)
    x = np.arange(len(all_steps), dtype=float)
    total_width = 0.8
    bar_width = total_width / max(1, n_groups)
    offsets = (np.arange(n_groups) - (n_groups - 1) / 2.0) * bar_width

    fig, ax = plt.subplots(figsize=(max(12, len(all_steps) * 1.1), 7.0))

    for idx, actor_key in enumerate(actor_keys):
        actor_df = grouped[actor_key].set_index(x_col)
        aligned = actor_df.reindex(all_steps).fillna(0.0).reset_index(drop=True)
        normalized = _normalize_rows(aligned, emotion_cols)
        bottom = np.zeros(len(all_steps), dtype=float)
        hatch = GROUP_HATCHES[idx % len(GROUP_HATCHES)]

        for emotion_col in emotion_cols:
            vals = normalized[emotion_col].to_numpy(dtype=float)
            ax.bar(
                x + offsets[idx],
                vals,
                width=bar_width * 0.95,
                bottom=bottom,
                color=EMOTION_COLORS.get(emotion_col, "#999999"),
                edgecolor="black",
                linewidth=0.2,
                hatch=hatch,
                label=emotion_col if idx == 0 else None,
            )
            bottom += vals

    ax.set_title(f"{run_dir.name} - Combined NRC Emotion Trajectory")
    ax.set_xlabel("Round" if x_col == "round" else "Turn Index")
    ax.set_ylabel("Normalized NRC emotion share")
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in all_steps])

    emotion_legend = ax.legend(
        title="Emotion",
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
    )
    ax.add_artist(emotion_legend)

    actor_handles = [
        Patch(facecolor="white", edgecolor="black", hatch=GROUP_HATCHES[i % len(GROUP_HATCHES)], label=actor)
        for i, actor in enumerate(actor_keys)
    ]
    ax.legend(
        handles=actor_handles,
        title="Model::Role",
        loc="upper left",
        bbox_to_anchor=(1.01, 0.45),
        frameon=False,
    )

    fig.tight_layout()
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "emotion_trajectory_combined.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return True


def plot_stance_vader_combined_for_run(run_dir: Path) -> bool:
    """Create a single unified plot with both stance (solid) and VADER (dashed) on same y-axis, all actors as colors."""
    features_path = run_dir / "utterance_features.csv"
    if not features_path.exists():
        return False

    df = pd.read_csv(features_path)
    if df.empty or "model_name" not in df.columns or "debate_role" not in df.columns:
        return False
    if "vader_compound" not in df.columns:
        return False

    x_col = "round" if "round" in df.columns and df["round"].notna().any() else "turn_index"
    if x_col not in df.columns:
        return False

    # Prepare data for stance (artifact label)
    work = df.copy()
    work[x_col] = pd.to_numeric(work[x_col], errors="coerce")
    if "stance" in work.columns:
        work["artifact_stance_numeric"] = work["stance"].apply(_stance_to_numeric)
    else:
        work["artifact_stance_numeric"] = np.nan

    work = work.dropna(subset=[x_col])
    if work.empty:
        return False

    # Build actor groups for both metrics
    actor_stance_groups: dict[str, pd.DataFrame] = {}
    actor_vader_groups: dict[str, pd.DataFrame] = {}
    for (model_name, debate_role), sub in work.groupby(["model_name", "debate_role"], dropna=False):
        actor_key = f"{str(model_name)}::{str(debate_role)}"
        # Stance
        stance_agg = sub[[x_col, "artifact_stance_numeric"]].groupby(x_col, as_index=False).mean(numeric_only=True).sort_values(x_col)
        if not stance_agg.empty:
            actor_stance_groups[actor_key] = stance_agg
        # VADER
        vader_agg = sub[[x_col, "vader_compound"]].groupby(x_col, as_index=False).mean(numeric_only=True).sort_values(x_col)
        if not vader_agg.empty:
            actor_vader_groups[actor_key] = vader_agg

    if not actor_stance_groups or not actor_vader_groups:
        return False

    all_steps = sorted(
        {
            int(step) if float(step).is_integer() else float(step)
            for actor_df in list(actor_stance_groups.values()) + list(actor_vader_groups.values())
            for step in actor_df[x_col].dropna().tolist()
        }
    )
    if not all_steps:
        return False

    actor_keys = sorted(set(actor_stance_groups.keys()) | set(actor_vader_groups.keys()))
    x = np.arange(len(all_steps), dtype=float)

    # Create single unified plot
    fig, ax = plt.subplots(figsize=(max(12, len(all_steps) * 1.1), 7.0))

    # Plot all actors with solid lines for stance, dashed lines for VADER
    for idx, actor_key in enumerate(actor_keys):
        color = ACTOR_LINE_COLORS[idx % len(ACTOR_LINE_COLORS)]
        
        # Stance: solid line
        if actor_key in actor_stance_groups:
            actor_df = actor_stance_groups[actor_key].set_index(x_col)
            aligned = actor_df.reindex(all_steps)
            y = aligned["artifact_stance_numeric"].to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=2.2,
                markersize=5.0,
                linestyle="-",
                label=f"{actor_key} (stance)",
                color=color,
            )
        
        # VADER: dashed line with same color
        if actor_key in actor_vader_groups:
            actor_df = actor_vader_groups[actor_key].set_index(x_col)
            aligned = actor_df.reindex(all_steps)
            y = aligned["vader_compound"].to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                marker="s",
                linewidth=2.2,
                markersize=4.5,
                linestyle="--",
                label=f"{actor_key} (VADER tone)",
                color=color,
                alpha=0.7,
            )

    ax.axhline(0.0, color="#374151", linewidth=1.2, linestyle=":", alpha=0.8)
    ax.set_ylabel("Position / Tone", fontsize=11, fontweight="bold")
    ax.set_ylim(-1.1, 1.1)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticklabels(["Oppose\n(Negative)", "-0.5", "Neutral\n(Neutral)", "0.5", "Support\n(Positive)"])
    ax.set_title(f"{run_dir.name} - Stance vs VADER Tone (All Actors Combined)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Round" if x_col == "round" else "Turn Index", fontsize=11, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in all_steps])
    
    ax.legend(
        title="Metric (Line Style)",
        title_fontsize=10,
        fontsize=9,
        frameon=True,
        fancybox=True,
        shadow=True,
        loc="best",
        ncol=2,
    )
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.grid(axis="x", alpha=0.15, linestyle=":")

    fig.tight_layout()
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / "stance_vader_combined.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return True





def main() -> None:
    parser = argparse.ArgumentParser(description="Create combined per-run NRC emotion trajectory plots.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "llm_view_analysis_outputs",
        help="Path to llm_view_analysis_outputs",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional run folder name (for example: run_20260208_102359)",
    )
    args = parser.parse_args()

    root = args.output_root
    if not root.exists():
        raise FileNotFoundError(f"Output root not found: {root}")

    if args.run_id:
        run_dirs = [root / args.run_id]
    else:
        run_dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("run_"))

    created_emotion = 0
    created_stance_vader = 0
    for run_dir in run_dirs:
        if not run_dir.exists():
            print(f"Skipping missing run folder: {run_dir}")
            continue
        ok_emotion = plot_combined_for_run(run_dir)
        ok_stance_vader = plot_stance_vader_combined_for_run(run_dir)

        if ok_emotion:
            created_emotion += 1
        if ok_stance_vader:
            created_stance_vader += 1

        if ok_emotion or ok_stance_vader:
            print(
                f"Created plots for {run_dir.name}: "
                f"emotion={'yes' if ok_emotion else 'no'}, "
                f"stance_vader={'yes' if ok_stance_vader else 'no'}"
            )
        else:
            print(f"Skipped {run_dir.name}: missing required columns or data")

    print(f"Done. Combined emotion plots created: {created_emotion}")
    print(f"Done. Combined stance+VADER plots created: {created_stance_vader}")


if __name__ == "__main__":
    main()
