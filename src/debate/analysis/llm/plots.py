from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#fcfcfd",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "--",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "font.size": 10,
        }
    )


def save_barplot(values: pd.Series, title: str, path: Path, ylabel: str, xlabel: str = "") -> None:
    if values.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = plt.cm.Blues([0.45 + 0.45 * (i / max(1, len(values) - 1)) for i in range(len(values))])
    values.plot(kind="bar", ax=ax, color=colors, edgecolor="#1f2937", linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for idx, value in enumerate(values.values):
        if pd.notna(value):
            ax.text(idx, float(value), f"{float(value):.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_lineplot(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    title: str,
    path: Path,
    xlabel: str = "Turn Index",
    ylabel: str = "Score",
) -> None:
    if df.empty or not y_cols:
        return
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    palette = plt.cm.Set2([i / max(1, len(y_cols)) for i in range(len(y_cols))])
    for i, col in enumerate(y_cols):
        if col in df.columns:
            ax.plot(
                df[x_col],
                df[col],
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                label=col,
                color=palette[i],
            )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_normalized_stacked_barplot(
    df: pd.DataFrame,
    x_col: str,
    y_cols: list[str],
    title: str,
    path: Path,
    xlabel: str = "Round",
    ylabel: str = "Normalized emotion share",
) -> None:
    if df.empty or x_col not in df.columns or not y_cols:
        return

    plot_df = df[[x_col] + [col for col in y_cols if col in df.columns]].copy()
    if plot_df.empty:
        return

    numeric_cols = [col for col in y_cols if col in plot_df.columns]
    if not numeric_cols:
        return

    plot_df[x_col] = pd.to_numeric(plot_df[x_col], errors="ignore")
    plot_df = plot_df.groupby(x_col, as_index=False)[numeric_cols].mean(numeric_only=True)
    if plot_df.empty:
        return

    plot_df = plot_df.sort_values(x_col).reset_index(drop=True)
    values = plot_df[numeric_cols].fillna(0.0).astype(float)
    row_sums = values.sum(axis=1).replace(0.0, np.nan)
    normalized = values.div(row_sums, axis=0).fillna(0.0)

    fig, ax = plt.subplots(figsize=(12.0, 6.2))
    x = np.arange(len(plot_df))
    bottom = np.zeros(len(plot_df), dtype=float)
    colors = plt.cm.Set3(np.linspace(0.05, 0.95, len(numeric_cols)))

    for idx, col in enumerate(numeric_cols):
        ax.bar(
            x,
            normalized[col].to_numpy(dtype=float),
            bottom=bottom,
            width=0.8,
            label=col.replace("emotion_", ""),
            color=colors[idx],
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += normalized[col].to_numpy(dtype=float)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.0)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in plot_df[x_col].tolist()], rotation=0)
    ax.legend(frameon=False, ncol=4, bbox_to_anchor=(0.5, -0.15), loc="upper center")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_boxplot(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    title: str,
    path: Path,
    xlabel: str,
    ylabel: str,
) -> None:
    if df.empty or group_col not in df.columns or value_col not in df.columns:
        return
    plot_df = df[[group_col, value_col]].dropna().copy()
    if plot_df.empty:
        return
    groups = sorted(plot_df[group_col].astype(str).unique().tolist())
    series_data = [
        plot_df.loc[plot_df[group_col].astype(str) == group, value_col].values for group in groups
    ]
    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    bp = ax.boxplot(series_data, tick_labels=groups, patch_artist=True, showmeans=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#bcd7f4")
        patch.set_edgecolor("#1f2937")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def save_heatmap(
    matrix: pd.DataFrame, title: str, path: Path, xlabel: str, ylabel: str, fmt: str = ".2f"
) -> None:
    if matrix.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, matrix.shape[1] * 1.1), max(5, matrix.shape[0] * 0.9)))
    im = ax.imshow(matrix.values, cmap="YlGnBu", aspect="auto")
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_xticklabels(matrix.columns, rotation=30, ha="right")
    ax.set_yticklabels(matrix.index)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            ax.text(c, r, format(matrix.iat[r, c], fmt), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    plt.tight_layout()
    plt.savefig(path, dpi=320, bbox_inches="tight")
    plt.close()


def make_wordcloud_or_bar(
    texts: list[str], title: str, outpath: Path, fallback_terms: pd.DataFrame
) -> dict:
    if texts:
        try:
            vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 1), min_df=1)
            X = vectorizer.fit_transform(texts)
            counts = np.asarray(X.sum(axis=0)).ravel()
            terms = vectorizer.get_feature_names_out()
            freqs = {term: int(count) for term, count in zip(terms, counts) if count > 0}
            if freqs:
                cloud = WordCloud(
                    width=1600, height=900, background_color="white", collocations=False
                )
                cloud.generate_from_frequencies(freqs)
                plt.figure(figsize=(12, 7))
                plt.imshow(cloud, interpolation="bilinear")
                plt.axis("off")
                plt.title(title)
                plt.tight_layout()
                plt.savefig(outpath, dpi=300, bbox_inches="tight")
                plt.close()
                return {
                    "method": "wordcloud",
                    "top_terms": sorted(freqs.items(), key=lambda x: -x[1])[:15],
                }
        except Exception:
            pass

    fallback = fallback_terms.copy() if not fallback_terms.empty else pd.DataFrame()
    if not fallback.empty:
        if "term_type" in fallback.columns:
            fallback = fallback[fallback["term_type"] == "unigram"].copy()
        fallback = fallback.head(15).sort_values("count", ascending=True)
        plt.figure(figsize=(12, 6))
        plt.barh(fallback["term"], fallback["count"])
        plt.title(f"{title} - Top Terms")
        plt.xlabel("Count")
        plt.tight_layout()
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close()
        return {"method": "bar_chart_fallback", "top_terms": fallback.to_dict("records")}

    return {"method": "unavailable", "top_terms": []}
