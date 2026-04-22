from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from .config import BASELINE_METRICS, BINARY_RATE_COLS, CORE_METRICS


def bootstrap_mean_ci(
    values: list[float], n_boot: int, ci: float = 0.95
) -> tuple[float, float, float]:
    clean = [float(v) for v in values if pd.notna(v)]
    if not clean:
        return 0.0, 0.0, 0.0
    if len(clean) == 1:
        x = clean[0]
        return x, x, x
    arr = np.array(clean, dtype=float)
    means = [
        float(np.random.choice(arr, size=len(arr), replace=True).mean()) for _ in range(n_boot)
    ]
    alpha = (1 - ci) / 2
    return float(arr.mean()), float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def bootstrap_gap_ci(
    a_values: list[float], b_values: list[float], n_boot: int, ci: float = 0.95
) -> tuple[float, float, float]:
    a = [float(v) for v in a_values if pd.notna(v)]
    b = [float(v) for v in b_values if pd.notna(v)]
    if not a or not b:
        return 0.0, 0.0, 0.0
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    gaps = []
    for _ in range(n_boot):
        a_sample = np.random.choice(a_arr, size=len(a_arr), replace=True)
        b_sample = np.random.choice(b_arr, size=len(b_arr), replace=True)
        gaps.append(float(a_sample.mean() - b_sample.mean()))
    observed = float(a_arr.mean() - b_arr.mean())
    alpha = (1 - ci) / 2
    return observed, float(np.quantile(gaps, alpha)), float(np.quantile(gaps, 1 - alpha))


def cohen_d(a: list[float], b: list[float]) -> float:
    a_clean = [float(v) for v in a if pd.notna(v)]
    b_clean = [float(v) for v in b if pd.notna(v)]
    if len(a_clean) < 2 or len(b_clean) < 2:
        return 0.0
    a_arr = np.array(a_clean, dtype=float)
    b_arr = np.array(b_clean, dtype=float)
    pooled = math.sqrt(max(0.0, ((a_arr.std() ** 2) + (b_arr.std() ** 2)) / 2.0))
    if pooled == 0:
        return 0.0
    return float((a_arr.mean() - b_arr.mean()) / pooled)


def group_summary(
    df: pd.DataFrame, group_cols: list[str], metrics: list[str] | None = None
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    numeric_cols = [col for col in (metrics or CORE_METRICS) if col in df.columns]
    if not numeric_cols:
        return pd.DataFrame()
    agg = df.groupby(group_cols)[numeric_cols].agg(["mean", "median", "std", "min", "max", "count"])
    agg.columns = [
        "_".join([str(part) for part in col if part]) for col in agg.columns.to_flat_index()
    ]
    agg = agg.reset_index()

    for col in BINARY_RATE_COLS:
        if col in df.columns:
            rate = (
                df.groupby(group_cols)[col]
                .mean()
                .reset_index()
                .rename(columns={col: f"{col}_rate"})
            )
            agg = agg.merge(rate, on=group_cols, how="left")

    total_words = (
        df.groupby(group_cols)["word_count"]
        .sum()
        .reset_index()
        .rename(columns={"word_count": "total_words"})
        if "word_count" in df.columns
        else pd.DataFrame()
    )
    utterance_count = df.groupby(group_cols).size().reset_index(name="utterance_count")
    if not total_words.empty:
        agg = agg.merge(total_words, on=group_cols, how="left")
    agg = agg.merge(utterance_count, on=group_cols, how="left")
    return agg


def group_summary_bootstrap(
    df: pd.DataFrame, group_cols: list[str], metrics: list[str], n_boot: int
) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    for values, group in df.groupby(group_cols):
        if not isinstance(values, tuple):
            values = (values,)
        base = {col: val for col, val in zip(group_cols, values)}
        for metric in metrics:
            if metric not in group.columns:
                continue
            mean_val, ci_low, ci_high = bootstrap_mean_ci(
                group[metric].dropna().tolist(), n_boot=n_boot
            )
            rows.append(
                {
                    **base,
                    "metric": metric,
                    "mean": mean_val,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n": int(group[metric].dropna().shape[0]),
                }
            )
    return pd.DataFrame(rows)


def trajectory_summary(
    df: pd.DataFrame, group_cols: list[str], step_col: str = "round"
) -> pd.DataFrame:
    if df.empty or step_col not in df.columns:
        return pd.DataFrame()
    metrics = [m for m in CORE_METRICS if m in df.columns]
    if not metrics:
        return pd.DataFrame()
    return df.groupby(group_cols + [step_col])[metrics].mean(numeric_only=True).reset_index()


def winner_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty or "winner_role" not in df.columns:
        return pd.DataFrame()
    winner_rows = df.dropna(subset=["winner_role"]).copy()
    if winner_rows.empty:
        return pd.DataFrame()
    debate_keys = ["run_id"] + [col for col in group_cols if col in winner_rows.columns]
    debate_level = winner_rows.groupby(debate_keys, as_index=False).first()
    winner_counts = (
        debate_level.groupby(group_cols + ["winner_role"], as_index=False)
        .size()
        .rename(columns={"size": "debate_count"})
    )
    totals = (
        debate_level.groupby(group_cols, as_index=False)
        .size()
        .rename(columns={"size": "total_debates"})
    )
    out = winner_counts.merge(totals, on=group_cols, how="left")
    out["winner_rate"] = out.apply(
        lambda row: (
            0.0
            if float(row["total_debates"]) == 0
            else float(row["debate_count"]) / float(row["total_debates"])
        ),
        axis=1,
    )
    return out.sort_values(
        group_cols + ["winner_rate", "debate_count"],
        ascending=[True] * len(group_cols) + [False, False],
    )


def role_contrast_summary(
    df: pd.DataFrame,
    metrics: list[str],
    n_boot: int,
    role_a: str = "proponent",
    role_b: str = "opponent",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    group_cols = group_cols or ["model_name"]
    if df.empty:
        return pd.DataFrame()
    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        metric_df = df[group_cols + ["debate_role", metric]].dropna().copy()
        if metric_df.empty:
            continue
        for values, group in metric_df.groupby(group_cols):
            if not isinstance(values, tuple):
                values = (values,)
            group_map = {col: val for col, val in zip(group_cols, values)}
            a_vals = group.loc[group["debate_role"] == role_a, metric].tolist()
            b_vals = group.loc[group["debate_role"] == role_b, metric].tolist()
            if not a_vals or not b_vals:
                continue
            gap, ci_low, ci_high = bootstrap_gap_ci(a_vals, b_vals, n_boot=n_boot)
            rows.append(
                {
                    **group_map,
                    "metric": metric,
                    f"{role_a}_mean": float(np.mean(a_vals)),
                    f"{role_b}_mean": float(np.mean(b_vals)),
                    f"{role_a}_n": int(len(a_vals)),
                    f"{role_b}_n": int(len(b_vals)),
                    "role_gap": gap,
                    "role_gap_ci_low": ci_low,
                    "role_gap_ci_high": ci_high,
                    "effect_size_d": cohen_d(a_vals, b_vals),
                }
            )
    return pd.DataFrame(rows)


def paired_run_role_gap(
    df: pd.DataFrame,
    metrics: list[str],
    n_boot: int,
    role_a: str = "proponent",
    role_b: str = "opponent",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    group_cols = group_cols or ["model_name"]
    if df.empty or "run_id" not in df.columns:
        return pd.DataFrame()
    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        sub = df[group_cols + ["run_id", "debate_role", metric]].dropna().copy()
        if sub.empty:
            continue
        per_run = sub.groupby(group_cols + ["run_id", "debate_role"], as_index=False)[metric].mean(
            numeric_only=True
        )
        pivot = per_run.pivot_table(
            index=group_cols + ["run_id"], columns="debate_role", values=metric, aggfunc="mean"
        )
        if role_a not in pivot.columns or role_b not in pivot.columns:
            continue
        pivot = pivot.dropna(subset=[role_a, role_b]).copy()
        if pivot.empty:
            continue
        pivot["role_gap"] = pivot[role_a] - pivot[role_b]
        pivot = pivot.reset_index()
        for values, group in pivot.groupby(group_cols):
            if not isinstance(values, tuple):
                values = (values,)
            group_map = {col: val for col, val in zip(group_cols, values)}
            gaps = group["role_gap"].dropna().astype(float)
            if gaps.empty:
                continue
            gap_mean, ci_low, ci_high = bootstrap_mean_ci(gaps.tolist(), n_boot=n_boot)
            rows.append(
                {
                    **group_map,
                    "metric": metric,
                    "paired_runs": int(len(gaps)),
                    "paired_gap_mean": gap_mean,
                    "paired_gap_ci_low": ci_low,
                    "paired_gap_ci_high": ci_high,
                    "paired_gap_median": float(gaps.median()),
                    "paired_gap_q10": float(gaps.quantile(0.10)),
                    "paired_gap_q90": float(gaps.quantile(0.90)),
                    "paired_gap_positive_rate": float((gaps > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def topic_role_balanced_comparison(
    df: pd.DataFrame, metrics: list[str], n_boot: int
) -> pd.DataFrame:
    if df.empty or "model_name" not in df.columns or df["model_name"].nunique() < 2:
        return pd.DataFrame()
    model_names = sorted(df["model_name"].dropna().astype(str).unique().tolist())
    if len(model_names) != 2:
        return pd.DataFrame()
    model_a, model_b = model_names

    rows = []
    for metric in metrics:
        if metric not in df.columns:
            continue
        grouped = (
            df.groupby(["topic_id", "debate_role", "model_name"], as_index=False)[metric]
            .mean(numeric_only=True)
            .pivot_table(index=["topic_id", "debate_role"], columns="model_name", values=metric)
            .dropna(subset=[model_a, model_b])
        )
        if grouped.empty:
            continue
        grouped = grouped.copy()
        grouped["gap_a_minus_b"] = grouped[model_a] - grouped[model_b]
        gaps = grouped["gap_a_minus_b"].astype(float).tolist()
        mean_gap, ci_low, ci_high = bootstrap_mean_ci(gaps, n_boot=n_boot)
        rows.append(
            {
                "metric": metric,
                "model_a": model_a,
                "model_b": model_b,
                "paired_topic_role_units": int(len(gaps)),
                "gap_mean_a_minus_b": mean_gap,
                "gap_ci_low": ci_low,
                "gap_ci_high": ci_high,
                "gap_median_a_minus_b": float(np.median(gaps)),
                "a_better_rate": float((np.array(gaps) > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def baseline_comparison(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for group_name, metrics in {
        "baseline": BASELINE_METRICS,
        "core": [m for m in CORE_METRICS if m in df.columns],
    }.items():
        for metric in metrics:
            if metric not in df.columns:
                continue
            for model_name, model_df in df.groupby("model_name"):
                vals = model_df[metric].dropna()
                rows.append(
                    {
                        "metric_family": group_name,
                        "metric": metric,
                        "model_name": model_name,
                        "mean": float(vals.mean()) if vals.shape[0] else np.nan,
                        "std": float(vals.std()) if vals.shape[0] > 1 else np.nan,
                        "n": int(vals.shape[0]),
                    }
                )
    return pd.DataFrame(rows)


def collect_error_cases(df: pd.DataFrame, examples_per_type: int = 20) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    rows = []
    if {"word_count", "evidence_density_per_1k_words"}.issubset(df.columns):
        sub = df.sort_values(
            ["word_count", "evidence_density_per_1k_words"], ascending=[False, True]
        ).head(examples_per_type)
        for _, row in sub.iterrows():
            rows.append(
                {
                    "error_type": "long_but_evidence_poor",
                    **row[
                        [
                            "run_id",
                            "model_name",
                            "debate_role",
                            "topic_id",
                            "turn_index",
                            "word_count",
                            "utterance",
                        ]
                    ].to_dict(),
                }
            )

    if {"confidence", "evidence_density_per_1k_words"}.issubset(df.columns):
        sub = (
            df.dropna(subset=["confidence"])
            .sort_values(["confidence", "evidence_density_per_1k_words"], ascending=[False, True])
            .head(examples_per_type)
        )
        for _, row in sub.iterrows():
            rows.append(
                {
                    "error_type": "high_confidence_low_evidence",
                    **row[
                        [
                            "run_id",
                            "model_name",
                            "debate_role",
                            "topic_id",
                            "turn_index",
                            "confidence",
                            "utterance",
                        ]
                    ].to_dict(),
                }
            )

    if {"rebuttal_density_per_1k_words", "semantic_similarity_prev_opponent_turn"}.issubset(
        df.columns
    ):
        sub = df.sort_values(
            ["rebuttal_density_per_1k_words", "semantic_similarity_prev_opponent_turn"],
            ascending=[False, True],
        ).head(examples_per_type)
        for _, row in sub.iterrows():
            rows.append(
                {
                    "error_type": "generic_rebuttal_marker",
                    **row[
                        [
                            "run_id",
                            "model_name",
                            "debate_role",
                            "topic_id",
                            "turn_index",
                            "utterance",
                        ]
                    ].to_dict(),
                }
            )

    if {"winner_confidence", "winner_role"}.issubset(df.columns):
        sub = df[df["winner_confidence"].isin(["low", "medium"])].head(examples_per_type)
        for _, row in sub.iterrows():
            rows.append(
                {
                    "error_type": "ambiguous_winner_extraction",
                    **row[
                        [
                            "run_id",
                            "model_name",
                            "debate_role",
                            "topic_id",
                            "turn_index",
                            "utterance",
                        ]
                    ].to_dict(),
                }
            )

    return pd.DataFrame(rows).drop_duplicates()


def representative_utterances(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame()
    sample_specs = {
        "highest_confidence": ("confidence", False),
        "most_evidence_dense": ("evidence_density_per_1k_words", False),
        "most_rebuttal_dense": ("rebuttal_density_per_1k_words", False),
        "most_hedged": ("hedge_density_per_1k_words", False),
        "most_assertive": ("assertive_density_per_1k_words", False),
        "most_moral_framed": ("moral_density_per_1k_words", False),
        "strongest_stance_proxy": ("stance_proxy_score", False),
        "weakest_stance_proxy": ("stance_proxy_score", True),
    }
    for (model_name, debate_role), group in df.groupby(["model_name", "debate_role"]):
        for label, (metric, ascending) in sample_specs.items():
            if metric not in group.columns:
                continue
            sample = group.sort_values([metric, "word_count"], ascending=[ascending, False]).head(1)
            if sample.empty:
                continue
            row = sample.iloc[0]
            rows.append(
                {
                    "model_name": model_name,
                    "debate_role": debate_role,
                    "sample_type": label,
                    "topic_id": row.get("topic_id"),
                    "topic_category": row.get("topic_category"),
                    "stage": row.get("stage"),
                    "round": row.get("round"),
                    "turn_index": row.get("turn_index"),
                    "confidence": row.get("confidence"),
                    "utterance": row.get("utterance"),
                }
            )
    return pd.DataFrame(rows)


def embedding_projection(all_df: pd.DataFrame, random_seed: int) -> pd.DataFrame:
    if all_df.empty or "embedding_vector_obj" not in all_df.columns:
        return pd.DataFrame()
    valid = all_df[all_df["embedding_vector_obj"].notna()].copy()
    if valid.empty:
        return pd.DataFrame()
    mat = np.vstack(valid["embedding_vector_obj"].to_list())
    mat = StandardScaler().fit_transform(mat)
    pca = PCA(n_components=2, random_state=random_seed)
    proj = pca.fit_transform(mat)
    valid["embed_pca_1"] = proj[:, 0]
    valid["embed_pca_2"] = proj[:, 1]
    return valid[
        [
            "run_id",
            "turn_index",
            "model_name",
            "debate_role",
            "topic_id",
            "stage",
            "embed_pca_1",
            "embed_pca_2",
        ]
    ]


def build_top_terms(texts: list[str], top_n: int, ngram_max: int = 3) -> pd.DataFrame:
    if not texts:
        return pd.DataFrame()
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, ngram_max), min_df=1)
    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:
        return pd.DataFrame()
    counts = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(vectorizer.get_feature_names_out())
    order = np.argsort(-counts)[:top_n]
    return pd.DataFrame(
        {
            "rank": list(range(1, len(order) + 1)),
            "term": terms[order],
            "term_type": [
                {1: "unigram", 2: "bigram", 3: "trigram"}.get(len(terms[idx].split()), "ngram")
                for idx in order
            ],
            "count": counts[order].astype(int),
        }
    )


def build_normalized_term_table(texts: list[str], top_n: int) -> pd.DataFrame:
    if not texts:
        return pd.DataFrame()
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 1), min_df=1)
    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:
        return pd.DataFrame()
    counts = np.asarray(X.sum(axis=0)).ravel()
    terms = np.array(vectorizer.get_feature_names_out())
    total_tokens = int(counts.sum())
    order = np.argsort(-counts)[:top_n]
    return pd.DataFrame(
        {
            "rank": list(range(1, len(order) + 1)),
            "term": terms[order],
            "count": counts[order].astype(int),
            "per_1k_tokens": [1000.0 * float(counts[idx]) / max(1, total_tokens) for idx in order],
            "total_tokens": total_tokens,
        }
    )


def build_distinctive_terms_for_group(
    all_df: pd.DataFrame, mask: pd.Series, top_n: int, min_df: int
) -> pd.DataFrame:
    target_texts = all_df.loc[mask, "utterance"].astype(str).tolist()
    other_texts = all_df.loc[~mask, "utterance"].astype(str).tolist()
    if not target_texts:
        return pd.DataFrame()

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=min_df)
    try:
        X = vectorizer.fit_transform(target_texts + other_texts)
    except ValueError:
        return pd.DataFrame()

    feature_names = np.array(vectorizer.get_feature_names_out())
    target_matrix = X[: len(target_texts)]
    other_matrix = X[len(target_texts) :]
    target_mean = np.asarray(target_matrix.mean(axis=0)).ravel()
    other_mean = (
        np.asarray(other_matrix.mean(axis=0)).ravel()
        if other_matrix.shape[0] > 0
        else np.zeros_like(target_mean)
    )
    diff = target_mean - other_mean
    order = np.argsort(-diff)[:top_n]
    return pd.DataFrame(
        {
            "rank": list(range(1, len(order) + 1)),
            "term": feature_names[order],
            "target_tfidf_mean": target_mean[order],
            "other_tfidf_mean": other_mean[order],
            "tfidf_gap": diff[order],
        }
    )
