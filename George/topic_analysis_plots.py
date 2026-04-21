#!/usr/bin/env python3
"""
Plotting script for the topic-level analysis outputs produced by topic_analysis_pipeline.py

What it creates
---------------
1) winners_per_topic.png
   Bar chart showing scientist wins, conspiracy wins, ties, and unknown per topic.

2) scientist_win_rate_per_topic.png
   Bar chart of scientist win rate per topic.

3) dominant_bert_emotion_per_topic.png
   Bar chart of the dominant BERT emotion for each topic.

4) bert_emotion_shares_per_topic.png
   Grouped bar chart of BERT predicted-emotion shares per topic.

5) uncertainty_by_topic_and_role.png
   Grouped bar chart comparing uncertainty rate for conspiracy vs scientist in each topic.

6) modality_by_topic_and_role.png
   Grouped bar chart comparing strong and weak modality rates for each topic and role.

7) top_words_by_topic_and_role.txt
   A readable text summary of the top words and top bigrams for each topic and role.

Usage
-----
python topic_analysis_plots.py --input_dir /path/to/topic_analysis_results --output_dir topic_analysis_plots
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", required=True, help="Directory produced by topic_analysis_pipeline.py")
    p.add_argument("--output_dir", required=True, help="Directory where plots will be written")
    return p.parse_args()


def save_bar_from_winner_topic(df: pd.DataFrame, out_path: Path):
    topics = df["topic_id"].astype(str).tolist()
    x = range(len(topics))

    plt.figure(figsize=(12, 6))
    width = 0.2
    plt.bar([i - 1.5 * width for i in x], df["scientist_wins"], width=width, label="scientist_wins")
    plt.bar([i - 0.5 * width for i in x], df["conspiracy_wins"], width=width, label="conspiracy_wins")
    plt.bar([i + 0.5 * width for i in x], df["ties"], width=width, label="ties")
    plt.bar([i + 1.5 * width for i in x], df["unknown"], width=width, label="unknown")

    plt.xticks(list(x), topics, rotation=45, ha="right")
    plt.ylabel("Number of debates")
    plt.title("Who usually wins in each topic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_scientist_win_rate(df: pd.DataFrame, out_path: Path):
    df = df.sort_values("scientist_win_rate", ascending=False)
    plt.figure(figsize=(11, 6))
    plt.bar(df["topic_id"].astype(str), df["scientist_win_rate"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Scientist win rate")
    plt.title("Scientist win rate per topic")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_dominant_emotion(df: pd.DataFrame, out_path: Path):
    plot_df = df[["topic_id", "dominant_bert_emotion"]].copy()
    cats = sorted(plot_df["dominant_bert_emotion"].dropna().unique().tolist())
    mapping = {c: i for i, c in enumerate(cats)}
    vals = plot_df["dominant_bert_emotion"].map(mapping)

    plt.figure(figsize=(11, 5))
    plt.bar(plot_df["topic_id"].astype(str), vals)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(list(mapping.values()), list(mapping.keys()))
    plt.ylabel("Dominant emotion")
    plt.title("Dominant BERT emotion per topic")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_bert_emotion_shares(df: pd.DataFrame, out_path: Path):
    emotion_cols = [c for c in df.columns if c.startswith("bert_share_")]
    if not emotion_cols:
        return

    topics = df["topic_id"].astype(str).tolist()
    x = list(range(len(topics)))
    width = 0.8 / max(len(emotion_cols), 1)

    plt.figure(figsize=(13, 6))
    for idx, col in enumerate(emotion_cols):
        pos = [i - 0.4 + width/2 + idx * width for i in x]
        plt.bar(pos, df[col].fillna(0.0), width=width, label=col.replace("bert_share_", ""))

    plt.xticks(x, topics, rotation=45, ha="right")
    plt.ylabel("Share of utterances")
    plt.title("BERT emotion shares per topic")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_uncertainty_by_role(df: pd.DataFrame, out_path: Path):
    pivot = df.pivot(index="topic_id", columns="role", values="uncertainty_rate").fillna(0.0)
    topics = pivot.index.astype(str).tolist()
    x = list(range(len(topics)))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar([i - width/2 for i in x], pivot.get("conspiracy", pd.Series([0]*len(pivot), index=pivot.index)), width=width, label="conspiracy")
    plt.bar([i + width/2 for i in x], pivot.get("scientist", pd.Series([0]*len(pivot), index=pivot.index)), width=width, label="scientist")
    plt.xticks(x, topics, rotation=45, ha="right")
    plt.ylabel("Uncertainty rate")
    plt.title("Uncertainty language by topic and role")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_modality_by_role(df: pd.DataFrame, out_path: Path):
    plot_df = df.copy()
    plot_df["topic_role"] = plot_df["topic_id"].astype(str) + " | " + plot_df["role"].astype(str)

    x = list(range(len(plot_df)))
    width = 0.35

    plt.figure(figsize=(14, 6))
    plt.bar([i - width/2 for i in x], plot_df["strong_modality_rate"].fillna(0.0), width=width, label="strong_modality_rate")
    plt.bar([i + width/2 for i in x], plot_df["weak_modality_rate"].fillna(0.0), width=width, label="weak_modality_rate")
    plt.xticks(x, plot_df["topic_role"], rotation=60, ha="right")
    plt.ylabel("Rate")
    plt.title("Strong vs weak modality by topic and role")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_top_words_text(df: pd.DataFrame, out_path: Path):
    lines = []
    for _, row in df.sort_values(["topic_id", "role"]).iterrows():
        lines.append(f"TOPIC: {row['topic_id']} | ROLE: {row['role']}")
        lines.append(f"Top words: {row.get('top_words', '')}")
        lines.append(f"Top bigrams: {row.get('top_bigrams', '')}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    winner_topic = pd.read_csv(input_dir / "winner_per_topic.csv")
    bert_topic = pd.read_csv(input_dir / "bert_emotions_per_topic.csv")
    lang_topic_role = pd.read_csv(input_dir / "language_by_topic_and_role.csv")

    save_bar_from_winner_topic(winner_topic, output_dir / "winners_per_topic.png")
    save_scientist_win_rate(winner_topic, output_dir / "scientist_win_rate_per_topic.png")
    save_dominant_emotion(bert_topic, output_dir / "dominant_bert_emotion_per_topic.png")
    save_bert_emotion_shares(bert_topic, output_dir / "bert_emotion_shares_per_topic.png")
    save_uncertainty_by_role(lang_topic_role, output_dir / "uncertainty_by_topic_and_role.png")
    save_modality_by_role(lang_topic_role, output_dir / "modality_by_topic_and_role.png")
    save_top_words_text(lang_topic_role, output_dir / "top_words_by_topic_and_role.txt")

    print("Plots written to:", output_dir)


if __name__ == "__main__":
    main()
