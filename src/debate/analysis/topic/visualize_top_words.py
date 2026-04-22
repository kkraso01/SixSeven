#!/usr/bin/env python3
"""
Create visualizations from top_words_by_topic_and_role.txt

What it creates
---------------
1) combined_topic_barplots/
   One horizontal bar chart per topic using combined frequencies from both roles.

2) role_split_topic_barplots/
   One stacked horizontal bar chart per topic showing conspiracy vs scientist contribution
   for the same top combined words.

3) topic_term_heatmap.png
   A heatmap across all topics using the most frequent combined words.

Usage
-----
python visualize_top_words.py \
  --input_txt /path/to/top_words_by_topic_and_role.txt \
  --output_dir top_words_visuals \
  --top_n 10 \
  --heatmap_terms 30
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def parse_counter_line(line: str) -> dict[str, int]:
    if ":" not in line:
        return {}
    line = line.split(":", 1)[1].strip()
    if not line:
        return {}
    items = {}
    parts = [p.strip() for p in line.split(",") if p.strip()]
    for part in parts:
        m = re.match(r"(.+):(\d+)$", part)
        if m:
            token = m.group(1).strip()
            count = int(m.group(2))
            items[token] = count
    return items


def parse_top_words_file(path: Path):
    data = defaultdict(dict)
    current_topic = None
    current_role = None

    lines = path.read_text(encoding="utf-8").splitlines()
    for raw in lines:
        line = raw.strip()
        if not line:
            continue

        if line.startswith("TOPIC:"):
            m = re.match(r"TOPIC:\s*(.+?)\s*\|\s*ROLE:\s*(.+)", line)
            if m:
                current_topic = m.group(1).strip()
                current_role = m.group(2).strip()
                if current_topic not in data:
                    data[current_topic] = {}
                if current_role not in data[current_topic]:
                    data[current_topic][current_role] = {"top_words": {}, "top_bigrams": {}}
        elif line.startswith("Top words:") and current_topic and current_role:
            data[current_topic][current_role]["top_words"] = parse_counter_line(line)
        elif line.startswith("Top bigrams:") and current_topic and current_role:
            data[current_topic][current_role]["top_bigrams"] = parse_counter_line(line)

    return data


def ensure_dirs(base: Path):
    (base / "combined_topic_barplots").mkdir(parents=True, exist_ok=True)
    (base / "role_split_topic_barplots").mkdir(parents=True, exist_ok=True)


def combined_topic_counts(topic_data: dict) -> dict[str, int]:
    combined = defaultdict(int)
    for role in topic_data:
        for token, count in topic_data[role].get("top_words", {}).items():
            combined[token] += count
    return dict(combined)


def save_combined_topic_barplot(topic: str, topic_data: dict, out_path: Path, top_n: int = 10):
    counts = combined_topic_counts(topic_data)
    if not counts:
        return
    s = pd.Series(counts).sort_values(ascending=True).tail(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(s.index.tolist(), s.values.tolist())
    plt.xlabel("Combined frequency")
    plt.ylabel("Word")
    plt.title(f"Top combined words for topic: {topic}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_role_split_topic_barplot(topic: str, topic_data: dict, out_path: Path, top_n: int = 10):
    combined = combined_topic_counts(topic_data)
    if not combined:
        return

    top_words = pd.Series(combined).sort_values(ascending=False).head(top_n).index.tolist()

    conspiracy = []
    scientist = []
    for word in top_words:
        conspiracy.append(topic_data.get("conspiracy", {}).get("top_words", {}).get(word, 0))
        scientist.append(topic_data.get("scientist", {}).get("top_words", {}).get(word, 0))

    plot_df = pd.DataFrame(
        {
            "word": top_words,
            "conspiracy": conspiracy,
            "scientist": scientist,
        }
    ).sort_values(["conspiracy", "scientist"], ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(plot_df["word"], plot_df["conspiracy"], label="conspiracy")
    plt.barh(plot_df["word"], plot_df["scientist"], left=plot_df["conspiracy"], label="scientist")
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.title(f"Top words for topic with role split: {topic}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def save_heatmap(all_data: dict, out_path: Path, heatmap_terms: int = 30):
    topic_combined = {}
    global_counts = defaultdict(int)

    for topic, topic_data in all_data.items():
        counts = combined_topic_counts(topic_data)
        topic_combined[topic] = counts
        for token, count in counts.items():
            global_counts[token] += count

    if not topic_combined:
        return

    top_terms = (
        pd.Series(global_counts).sort_values(ascending=False).head(heatmap_terms).index.tolist()
    )

    rows = []
    for topic, counts in topic_combined.items():
        row = {"topic": topic}
        for term in top_terms:
            row[term] = counts.get(term, 0)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("topic")
    df = df.sort_index()

    plt.figure(figsize=(max(12, len(top_terms) * 0.45), max(7, len(df) * 0.45)))
    plt.imshow(df.values, aspect="auto")
    plt.colorbar(label="Frequency")
    plt.xticks(range(len(df.columns)), df.columns, rotation=75, ha="right")
    plt.yticks(range(len(df.index)), df.index)
    plt.title("Topic-term heatmap from top words")
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

    df.to_csv(out_path.with_suffix(".csv"))


def save_summary_readme(all_data: dict, out_path: Path):
    lines = []
    lines.append("How to use these visuals")
    lines.append("")
    lines.append(
        "1. combined_topic_barplots: use these when your priority is the topic as a whole."
    )
    lines.append(
        "2. role_split_topic_barplots: use these when you want to add a secondary note about role differences."
    )
    lines.append(
        "3. topic_term_heatmap.png: use this as the overall lexical overview across all topics."
    )
    lines.append("")
    lines.append("Suggested report usage:")
    lines.append("- Put the heatmap first as the overview figure.")
    lines.append("- Then show 2 to 4 topic barplots for the most interesting topics.")
    lines.append(
        "- If needed, follow each barplot with the role-split version for one short comparison."
    )
    lines.append("")
    lines.append("Topics found:")
    for topic in sorted(all_data.keys()):
        roles = ", ".join(sorted(all_data[topic].keys()))
        lines.append(f"- {topic}: {roles}")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_txt",
        default="results/analysis/topic_analysis/plots/top_words_by_topic_and_role.txt",
        help="Path to top_words_by_topic_and_role.txt",
    )
    parser.add_argument(
        "--output_dir",
        default="results/analysis/topic_analysis/top_words_visuals",
        help="Directory for generated visuals",
    )
    parser.add_argument("--top_n", type=int, default=10, help="Top words to show per topic")
    parser.add_argument(
        "--heatmap_terms", type=int, default=30, help="Number of terms to show in heatmap"
    )
    args = parser.parse_args(argv)

    input_path = Path(args.input_txt)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_dirs(output_dir)

    all_data = parse_top_words_file(input_path)

    for topic, topic_data in all_data.items():
        safe_topic = re.sub(r"[^A-Za-z0-9_\-]+", "_", topic)
        save_combined_topic_barplot(
            topic,
            topic_data,
            output_dir / "combined_topic_barplots" / f"{safe_topic}_combined.png",
            top_n=args.top_n,
        )
        save_role_split_topic_barplot(
            topic,
            topic_data,
            output_dir / "role_split_topic_barplots" / f"{safe_topic}_role_split.png",
            top_n=args.top_n,
        )

    save_heatmap(all_data, output_dir / "topic_term_heatmap.png", heatmap_terms=args.heatmap_terms)
    save_summary_readme(all_data, output_dir / "README_visuals.txt")

    print(f"Done. Visuals saved to: {output_dir}")


if __name__ == "__main__":
    main()
