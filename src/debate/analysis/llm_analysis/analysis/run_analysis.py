from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CORE_METRICS,
    METRICS_FOR_COMPARISON_PLOTS,
    METRICS_FOR_TRAJECTORY_PLOTS,
    ROLE_TO_MODEL_KEY,
    default_config,
)
from .debate_metrics import (
    build_stance_change_events,
    build_turning_point_events,
    compute_interaction_features,
    compute_stage_labels,
    infer_winner_from_final_report,
    summarise_stance_change,
    summarise_turning_points,
)
from .io_utils import (
    canonical_role,
    ensure_dir,
    normalize_name,
    persist_object_columns,
    read_csv_if_exists,
    read_json_if_exists,
    read_topics_catalog,
    rehydrate_cached_object_columns,
    save_frames,
    write_json,
)
from .plots import (
    configure_plot_style,
    make_wordcloud_or_bar,
    save_barplot,
    save_boxplot,
    save_heatmap,
    save_lineplot,
    save_normalized_stacked_barplot,
)
from .summaries import (
    baseline_comparison,
    build_distinctive_terms_for_group,
    build_normalized_term_table,
    build_top_terms,
    collect_error_cases,
    embedding_projection,
    group_summary,
    group_summary_bootstrap,
    paired_run_role_gap,
    representative_utterances,
    role_contrast_summary,
    topic_role_balanced_comparison,
    trajectory_summary,
    winner_summary,
)
from .text_features import analyze_utterance, embedding_vector, load_nlp_resources


CONFIG = default_config()
NRC_TRAJECTORY_EMOTIONS = ["anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust"]


def _emotion_trajectory_columns(df: pd.DataFrame) -> list[str]:
    return [f"emotion_{emotion}" for emotion in NRC_TRAJECTORY_EMOTIONS if f"emotion_{emotion}" in df.columns]


def _save_run_emotion_trajectory(
    group_df: pd.DataFrame,
    run_output_dir: Path,
    model_name: str,
    debate_role: str,
) -> None:
    emotion_cols = _emotion_trajectory_columns(group_df)
    if not emotion_cols:
        return

    working_df = group_df.copy()
    x_col = "round" if working_df.get("round") is not None and working_df["round"].notna().any() else "turn_index"
    if x_col not in working_df.columns:
        x_col = "turn_index"

    working_df[x_col] = pd.to_numeric(working_df[x_col], errors="coerce")
    working_df = working_df.dropna(subset=[x_col])
    if working_df.empty:
        return

    plot_df = working_df[[x_col] + emotion_cols].copy()
    plot_df = plot_df.groupby(x_col, as_index=False)[emotion_cols].mean(numeric_only=True).sort_values(x_col)
    if plot_df.empty:
        return

    safe_model = normalize_name(model_name)
    safe_role = normalize_name(debate_role)
    title = f"{model_name} - {debate_role} NRC emotion profile across rounds"
    plot_path = run_output_dir / "plots" / f"emotion_trajectory_{safe_model}_{safe_role}.png"
    csv_path = run_output_dir / f"emotion_trajectory_{safe_model}_{safe_role}.csv"
    plot_df.to_csv(csv_path, index=False)
    save_normalized_stacked_barplot(
        plot_df,
        x_col=x_col,
        y_cols=emotion_cols,
        title=title,
        path=plot_path,
        xlabel="Round" if x_col == "round" else "Turn Index",
        ylabel="Normalized NRC emotion share",
    )


def load_run_metadata(run_dir: Path, topic_catalog: dict[str, dict]) -> dict:
    run_config = read_json_if_exists(run_dir / "run_config.json") or {}
    experiment_metadata = read_json_if_exists(run_dir / "experiment_metadata.json") or {}
    final_report = read_json_if_exists(run_dir / "final_report.json") or {}
    metrics = read_csv_if_exists(run_dir / "metrics.csv")
    if metrics is None:
        metrics = pd.DataFrame()

    topic_id = experiment_metadata.get("topic_id") or final_report.get("topic_id")
    topic_info = topic_catalog.get(str(topic_id), {}) if topic_id is not None else {}
    models = run_config.get("models", {}) or {}

    return {
        "run_config": run_config,
        "experiment_metadata": experiment_metadata,
        "final_report": final_report,
        "metrics": metrics,
        "topic_id": topic_id,
        "topic_info": topic_info,
        "models": models,
    }


def compute_model_name(models: dict, speaker_role: str) -> str:
    role = canonical_role(speaker_role)
    model_key = ROLE_TO_MODEL_KEY.get(role, role)
    return str(models.get(model_key, model_key))


def load_cached_run_outputs(run_dir: Path, output_root: Path) -> tuple[pd.DataFrame | None, dict | None]:
    run_output_dir = output_root / run_dir.name
    features_path = run_output_dir / "utterance_features.csv"
    metadata_path = run_output_dir / "analysis_metadata.json"
    if not features_path.exists():
        return None, None

    df_cached = read_csv_if_exists(features_path)
    if df_cached is None or df_cached.empty:
        return None, None

    df_cached = rehydrate_cached_object_columns(df_cached)
    metadata = read_json_if_exists(metadata_path) or {"run_id": run_dir.name, "run_dir": str(run_dir)}
    if metadata.get("analysis_schema_version") != CONFIG.analysis_schema_version:
        return None, None

    metadata["from_cache"] = True
    metadata["skipped_recompute"] = True
    return df_cached, metadata


def enrich_run_frame(run_dir: Path, run_meta: dict, resources) -> tuple[pd.DataFrame, dict]:
    debate_log = read_csv_if_exists(run_dir / "debate_log.csv")
    if debate_log is None or debate_log.empty:
        return pd.DataFrame(), {"skipped": True, "reason": "debate_log_missing"}
    if "utterance" not in debate_log.columns:
        raise ValueError(f"{run_dir / 'debate_log.csv'} does not contain an 'utterance' column.")

    df = debate_log.copy().reset_index(drop=True)
    df["turn_index"] = range(1, len(df) + 1)
    df["speaker_role"] = df.get("speaker_role", "unknown")
    df["speaker_role"] = df["speaker_role"].fillna("unknown").astype(str).apply(canonical_role)
    df["debate_role"] = df["speaker_role"].astype(str)
    df["model_name"] = df["speaker_role"].apply(lambda role: compute_model_name(run_meta["models"], role))

    if "round" not in df.columns:
        df["round"] = pd.NA
    if "confidence" not in df.columns:
        df["confidence"] = pd.NA
    df["confidence"] = pd.to_numeric(df["confidence"], errors="coerce")

    experiment_metadata = run_meta["experiment_metadata"]
    final_report = run_meta["final_report"]
    topic_info = run_meta["topic_info"]
    topic_motion = topic_info.get("motion") or experiment_metadata.get("motion")

    features = [
        analyze_utterance(resources, text, role, claim=topic_motion)
        for text, role in zip(df["utterance"].astype(str), df["speaker_role"].astype(str))
    ]
    df = pd.concat([df, pd.DataFrame(features)], axis=1)
    df["embedding_vector_obj"] = df["utterance"].astype(str).apply(lambda x: embedding_vector(resources.embedding_model, x))

    winner_info = infer_winner_from_final_report(final_report)

    df["run_dir"] = str(run_dir)
    df["run_id"] = run_dir.name
    df["topic_id"] = run_meta.get("topic_id")
    df["topic_category"] = topic_info.get("category") or experiment_metadata.get("topic_category")
    df["topic_title"] = topic_info.get("topic") or experiment_metadata.get("topic_description") or experiment_metadata.get("topic_title")
    df["topic_motion"] = topic_motion
    df["winner_inferred"] = winner_info.get("winner_inferred")
    df["winner_role"] = winner_info.get("winner_role")
    df["winner_source"] = winner_info.get("winner_source")
    df["winner_confidence"] = winner_info.get("winner_confidence")
    df["stage"] = compute_stage_labels(df)

    ordered = df.sort_values(["speaker_role", "turn_index"]).copy()
    ordered["confidence_delta_within_role"] = ordered.groupby("speaker_role")["confidence"].diff()
    ordered["previous_utterance_count_within_role"] = ordered.groupby("speaker_role").cumcount()
    ordered["stance_proxy_delta_within_role"] = ordered.groupby("speaker_role")["stance_proxy_score"].diff()
    
    if "role_alignment_score" in ordered.columns:
        ordered["role_alignment_score_delta_within_role"] = ordered.groupby("speaker_role")["role_alignment_score"].diff()
    else:
        ordered["role_alignment_score_delta_within_role"] = pd.NA

    if "role_alignment_label" in ordered.columns:
        ordered["role_misalignment_event"] = (ordered["role_alignment_label"] == "misaligned").astype(int)
    else:
        ordered["role_misalignment_event"] = 0

    if "predicted_stance_label" in ordered.columns:
        prev_stance = ordered.groupby("speaker_role")["predicted_stance_label"].shift()
        ordered["stance_flip_event"] = (
            ((prev_stance == "support") & (ordered["predicted_stance_label"] == "oppose")) |
            ((prev_stance == "oppose") & (ordered["predicted_stance_label"] == "support"))
        ).astype(int)
    else:
        ordered["stance_flip_event"] = 0
    
    df = df.merge(
        ordered[[
            "turn_index", "speaker_role", "confidence_delta_within_role",
            "previous_utterance_count_within_role", "stance_proxy_delta_within_role",
            "role_alignment_score_delta_within_role", "role_misalignment_event", "stance_flip_event",
        ]],
        on=["turn_index", "speaker_role"],
        how="left",
    )

    df = compute_interaction_features(df, resources.stop_words)

    stance_span = (
        df.groupby(["run_id", "speaker_role"], as_index=False)["stance_proxy_score"]
        .agg(["min", "max", "mean", "std"]).reset_index()
    )
    stance_span["stance_proxy_span_within_role"] = stance_span["max"] - stance_span["min"]
    df = df.merge(
        stance_span[["run_id", "speaker_role", "stance_proxy_span_within_role"]],
        on=["run_id", "speaker_role"],
        how="left"
    )

    return df, {
        "skipped": False,
        "topic_info": topic_info,
        "final_report": final_report,
        "experiment_metadata": experiment_metadata,
        "run_config": run_meta["run_config"],
        "winner_info": winner_info,
    }


def write_run_outputs(df: pd.DataFrame, run_dir: Path, run_meta: dict, output_root: Path) -> dict:
    run_output_dir = output_root / run_dir.name
    plots_dir = run_output_dir / "plots"
    ensure_dir(run_output_dir)
    ensure_dir(plots_dir)

    if run_meta.get("skipped"):
        return {"run_id": run_dir.name, "skipped": True}

    experiment_metadata = run_meta["experiment_metadata"]
    run_config = run_meta["run_config"]
    topic_info = run_meta["topic_info"]
    winner_info = run_meta["winner_info"]

    analysis_summary = {
        "run_id": run_dir.name,
        "run_dir": str(run_dir),
        "analysis_schema_version": CONFIG.analysis_schema_version,
        "row_count": int(len(df)),
        "topic_id": run_meta.get("topic_id"),
        "topic_category": topic_info.get("category") or experiment_metadata.get("topic_category"),
        "topic_title": topic_info.get("topic") or experiment_metadata.get("topic_description") or experiment_metadata.get("topic_title"),
        "topic_motion": topic_info.get("motion") or experiment_metadata.get("motion"),
        "models": run_config.get("models", {}) or {},
        "winner_info": winner_info,
        "random_seed": CONFIG.random_seed,
        "bootstrap_samples": CONFIG.bootstrap_samples,
        "metric_families": ["baseline", "core", "interaction", "outcome", "error_analysis", "stance_change", "turning_points", "novelty", "emotion_trajectory"],
    }

    persist_object_columns(df).to_csv(run_output_dir / "utterance_features.csv", index=False)

    model_summary = group_summary(df, ["model_name"])
    role_summary_df = group_summary(df, ["debate_role"])
    model_role_summary = group_summary(df, ["model_name", "debate_role"])
    stage_summary = group_summary(df, ["model_name", "debate_role", "stage"])
    topic_summary = group_summary(df, ["topic_id", "topic_category", "model_name", "debate_role"])
    bootstrap_summary = group_summary_bootstrap(df, ["model_name", "debate_role"], [m for m in CORE_METRICS if m in df.columns], n_boot=CONFIG.bootstrap_samples)
    role_contrast = role_contrast_summary(df, [m for m in CORE_METRICS if m in df.columns], n_boot=CONFIG.bootstrap_samples, group_cols=["model_name"])
    paired_role = paired_run_role_gap(df, [m for m in CORE_METRICS if m in df.columns], n_boot=CONFIG.bootstrap_samples, group_cols=["model_name"])
    trajectory_round = trajectory_summary(df, ["model_name", "debate_role"], step_col="round")
    trajectory_stage = trajectory_summary(df, ["model_name", "debate_role"], step_col="stage")
    winner_summary_df = winner_summary(df, ["model_name", "debate_role"])
    errors = collect_error_cases(df, examples_per_type=CONFIG.error_examples_per_type)
    baseline = baseline_comparison(df)
    representative = representative_utterances(df)
    stance_events = build_stance_change_events(df, ["model_name", "debate_role"])
    stance_summary = summarise_stance_change(stance_events, ["model_name", "debate_role"])
    turning_events = build_turning_point_events(df, ["model_name", "debate_role"], threshold=CONFIG.turning_point_delta_threshold)
    turning_summary = summarise_turning_points(turning_events, ["model_name", "debate_role"])

    save_frames(
        run_output_dir,
        {
            "model_summary.csv": model_summary,
            "role_summary.csv": role_summary_df,
            "model_role_summary.csv": model_role_summary,
            "stage_summary.csv": stage_summary,
            "topic_summary.csv": topic_summary,
            "bootstrap_summary.csv": bootstrap_summary,
            "role_contrast.csv": role_contrast,
            "paired_run_role_gap.csv": paired_role,
            "trajectory_by_model_role_round.csv": trajectory_round,
            "trajectory_by_model_role_stage.csv": trajectory_stage,
            "winner_summary.csv": winner_summary_df,
            "baseline_summary.csv": baseline,
            "error_analysis_candidates.csv": errors,
            "representative_utterances.csv": representative,
            "stance_change_events.csv": stance_events,
            "stance_change_summary.csv": stance_summary,
            "turning_point_events.csv": turning_events,
            "turning_point_summary.csv": turning_summary,
        },
    )

    term_rows = []
    norm_rows = []
    distinct_rows = []
    wordcloud_meta = []
    for (model_name, debate_role), group in df.groupby(["model_name", "debate_role"]):
        texts = group["utterance"].astype(str).tolist()
        terms = build_top_terms(texts, top_n=CONFIG.top_n_terms)
        norm = build_normalized_term_table(texts, top_n=CONFIG.top_n_terms)
        mask = (df["model_name"] == model_name) & (df["debate_role"] == debate_role)
        distinct = build_distinctive_terms_for_group(
            df,
            mask,
            top_n=CONFIG.top_n_distinctive_terms,
            min_df=CONFIG.min_df_distinctive_terms,
        )
        if not terms.empty:
            tmp = terms.copy()
            tmp.insert(0, "group_name", f"{model_name}::{debate_role}")
            term_rows.append(tmp)
        if not norm.empty:
            tmp = norm.copy()
            tmp.insert(0, "group_name", f"{model_name}::{debate_role}")
            norm_rows.append(tmp)
        if not distinct.empty:
            tmp = distinct.copy()
            tmp.insert(0, "group_name", f"{model_name}::{debate_role}")
            distinct_rows.append(tmp)
        wc = make_wordcloud_or_bar(
            texts,
            f"{model_name} - {debate_role}",
            plots_dir / f"wordcloud_{normalize_name(model_name)}_{normalize_name(debate_role)}.png",
            terms,
        )
        wc.update({"group_name": f"{model_name}::{debate_role}"})
        wordcloud_meta.append(wc)

        _save_run_emotion_trajectory(group, run_output_dir, model_name, debate_role)

    if term_rows:
        pd.concat(term_rows, ignore_index=True).to_csv(run_output_dir / "top_terms.csv", index=False)
    if norm_rows:
        pd.concat(norm_rows, ignore_index=True).to_csv(run_output_dir / "normalized_top_terms.csv", index=False)
    if distinct_rows:
        pd.concat(distinct_rows, ignore_index=True).to_csv(run_output_dir / "distinctive_terms.csv", index=False)
    if wordcloud_meta:
        write_json(run_output_dir / "wordcloud_metadata.json", wordcloud_meta)

    write_json(run_output_dir / "analysis_metadata.json", analysis_summary)
    return analysis_summary


def write_model_folders(all_df: pd.DataFrame, output_root: Path) -> None:
    models_root = output_root / "models"
    ensure_dir(models_root)

    for model_name, model_df in all_df.groupby("model_name"):
        model_dir = models_root / normalize_name(model_name)
        plots_dir = model_dir / "plots"
        ensure_dir(model_dir)
        ensure_dir(plots_dir)

        persist_object_columns(model_df).to_csv(model_dir / "utterance_features.csv", index=False)

        role_summary_df = group_summary(model_df, ["debate_role"])
        stage_summary = group_summary(model_df, ["debate_role", "stage"])
        topic_summary = group_summary(model_df, ["topic_id", "topic_category", "debate_role"])
        trajectory_round = trajectory_summary(model_df, ["debate_role"], step_col="round")
        trajectory_stage = trajectory_summary(model_df, ["debate_role"], step_col="stage")
        winner_summary_df = winner_summary(model_df, ["debate_role"])
        role_contrast = role_contrast_summary(model_df, [m for m in CORE_METRICS if m in model_df.columns], n_boot=CONFIG.bootstrap_samples, group_cols=["model_name"])
        paired_role = paired_run_role_gap(model_df, [m for m in CORE_METRICS if m in model_df.columns], n_boot=CONFIG.bootstrap_samples, group_cols=["model_name"])
        bootstrap = group_summary_bootstrap(model_df, ["debate_role", "stage"], [m for m in CORE_METRICS if m in model_df.columns], n_boot=CONFIG.bootstrap_samples)
        baseline = baseline_comparison(model_df)
        errors = collect_error_cases(model_df, examples_per_type=CONFIG.error_examples_per_type)
        stance_events = build_stance_change_events(model_df, ["debate_role"])
        stance_summary = summarise_stance_change(stance_events, ["debate_role"])
        turning_events = build_turning_point_events(model_df, ["debate_role"], threshold=CONFIG.turning_point_delta_threshold)
        turning_summary = summarise_turning_points(turning_events, ["debate_role"])

        save_frames(
            model_dir,
            {
                "role_summary.csv": role_summary_df,
                "stage_summary.csv": stage_summary,
                "topic_summary.csv": topic_summary,
                "trajectory_by_role_round.csv": trajectory_round,
                "trajectory_by_role_stage.csv": trajectory_stage,
                "winner_summary_by_role.csv": winner_summary_df,
                "role_contrast_metrics.csv": role_contrast,
                "paired_run_role_gap.csv": paired_role,
                "bootstrap_summary.csv": bootstrap,
                "baseline_summary.csv": baseline,
                "error_analysis_candidates.csv": errors,
                "stance_change_events.csv": stance_events,
                "stance_change_summary.csv": stance_summary,
                "turning_point_events.csv": turning_events,
                "turning_point_summary.csv": turning_summary,
            },
        )

        texts = model_df["utterance"].astype(str).tolist()
        terms = build_top_terms(texts, top_n=CONFIG.top_n_terms)
        norm_terms = build_normalized_term_table(texts, top_n=CONFIG.top_n_terms)
        if not terms.empty:
            terms.to_csv(model_dir / "top_terms.csv", index=False)
        if not norm_terms.empty:
            norm_terms.to_csv(model_dir / "normalized_top_terms.csv", index=False)

        representative = representative_utterances(model_df)
        if not representative.empty:
            representative.to_csv(model_dir / "representative_utterances.csv", index=False)

        wordcloud_info = make_wordcloud_or_bar(
            texts,
            f"{model_name} - Overall Vocabulary",
            plots_dir / "wordcloud_overall.png",
            terms,
        )
        write_json(model_dir / "wordcloud_metadata.json", wordcloud_info)

        for metric in METRICS_FOR_COMPARISON_PLOTS:
            if metric in model_df.columns:
                role_metric = model_df.groupby("debate_role")[metric].mean(numeric_only=True).sort_values(ascending=False)
                if not role_metric.empty:
                    save_boxplot(model_df, "debate_role", metric, f"{model_name} - Distribution by role ({metric})", plots_dir / f"role_{metric}_boxplot.png", "Role", metric)

        for metric in METRICS_FOR_TRAJECTORY_PLOTS:
            if metric in trajectory_round.columns and not trajectory_round.empty:
                pivot = trajectory_round.pivot(index="round", columns="debate_role", values=metric).reset_index()
                if not pivot.empty:
                    save_lineplot(pivot, "round", [c for c in pivot.columns if c != "round"], f"{model_name} - {metric} through rounds", plots_dir / f"trajectory_round_{metric}.png", "Round", metric)



def write_aggregate_outputs(all_df: pd.DataFrame, run_summaries: list[dict], output_root: Path) -> None:
    ensure_dir(output_root)
    persist_object_columns(all_df).to_csv(output_root / "all_utterance_features.csv", index=False)
    if run_summaries:
        pd.DataFrame(run_summaries).to_csv(output_root / "run_summaries.csv", index=False)

    write_model_folders(all_df, output_root)

    model_summary = group_summary(all_df, ["model_name"])
    role_summary_df = group_summary(all_df, ["debate_role"])
    model_role_summary = group_summary(all_df, ["model_name", "debate_role"])
    model_role_stage_summary = group_summary(all_df, ["model_name", "debate_role", "stage"])
    topic_summary = group_summary(all_df, ["topic_id", "topic_category"])
    topic_model_summary = group_summary(all_df, ["topic_id", "topic_category", "model_name", "debate_role"])
    model_bootstrap = group_summary_bootstrap(all_df, ["model_name", "debate_role"], [m for m in CORE_METRICS if m in all_df.columns], n_boot=CONFIG.bootstrap_samples)
    model_winner_summary = winner_summary(all_df, ["model_name"])
    model_role_winner_summary = winner_summary(all_df, ["model_name", "debate_role"])
    model_role_contrast = role_contrast_summary(all_df, [m for m in CORE_METRICS if m in all_df.columns], n_boot=CONFIG.bootstrap_samples, group_cols=["model_name"])
    model_role_paired = paired_run_role_gap(all_df, [m for m in CORE_METRICS if m in all_df.columns], n_boot=CONFIG.bootstrap_samples, group_cols=["model_name"])
    topic_balanced = topic_role_balanced_comparison(all_df, [m for m in CORE_METRICS if m in all_df.columns], n_boot=CONFIG.bootstrap_samples)
    baseline = baseline_comparison(all_df)
    errors = collect_error_cases(all_df, examples_per_type=CONFIG.error_examples_per_type)
    stance_events = build_stance_change_events(all_df, ["model_name", "debate_role"])
    stance_summary = summarise_stance_change(stance_events, ["model_name", "debate_role"])
    turning_events = build_turning_point_events(all_df, ["model_name", "debate_role"], threshold=CONFIG.turning_point_delta_threshold)
    turning_summary = summarise_turning_points(turning_events, ["model_name", "debate_role"])

    save_frames(
        output_root,
        {
            "aggregate_model_summary.csv": model_summary,
            "aggregate_role_summary.csv": role_summary_df,
            "aggregate_model_role_summary.csv": model_role_summary,
            "aggregate_model_role_stage_summary.csv": model_role_stage_summary,
            "aggregate_topic_summary.csv": topic_summary,
            "aggregate_topic_model_summary.csv": topic_model_summary,
            "aggregate_model_bootstrap_summary.csv": model_bootstrap,
            "aggregate_model_winner_summary.csv": model_winner_summary,
            "aggregate_model_role_winner_summary.csv": model_role_winner_summary,
            "aggregate_model_role_contrast.csv": model_role_contrast,
            "aggregate_model_role_paired_run_gap.csv": model_role_paired,
            "aggregate_topic_role_balanced_comparison.csv": topic_balanced,
            "aggregate_baseline_summary.csv": baseline,
            "aggregate_error_analysis_candidates.csv": errors,
            "aggregate_stance_change_events.csv": stance_events,
            "aggregate_stance_change_summary.csv": stance_summary,
            "aggregate_turning_point_events.csv": turning_events,
            "aggregate_turning_point_summary.csv": turning_summary,
        },
    )

    trajectory_round = trajectory_summary(all_df, ["model_name", "debate_role"], step_col="round")
    trajectory_stage = trajectory_summary(all_df, ["model_name", "debate_role"], step_col="stage")
    if not trajectory_round.empty:
        trajectory_round.to_csv(output_root / "aggregate_trajectory_by_model_role_round.csv", index=False)
    if not trajectory_stage.empty:
        trajectory_stage.to_csv(output_root / "aggregate_trajectory_by_model_role_stage.csv", index=False)

    agg_term_rows = []
    agg_norm_rows = []
    agg_distinct_rows = []
    for group_type, group_cols in [
        ("model", ["model_name"]),
        ("role", ["debate_role"]),
        ("model_role", ["model_name", "debate_role"]),
        ("topic", ["topic_id"]),
    ]:
        for values, group in all_df.groupby(group_cols):
            if isinstance(values, tuple):
                group_name = "::".join(str(v) for v in values)
                mask = np.logical_and.reduce([all_df[col] == val for col, val in zip(group_cols, values)])
            else:
                group_name = str(values)
                mask = all_df[group_cols[0]] == values

            texts = group["utterance"].astype(str).tolist()
            terms = build_top_terms(texts, top_n=CONFIG.top_n_terms)
            norm_terms = build_normalized_term_table(texts, top_n=CONFIG.top_n_terms)
            distinct_terms = build_distinctive_terms_for_group(
                all_df,
                pd.Series(mask, index=all_df.index),
                top_n=CONFIG.top_n_distinctive_terms,
                min_df=CONFIG.min_df_distinctive_terms,
            )

            if not terms.empty:
                tmp = terms.copy()
                tmp.insert(0, "group_type", group_type)
                tmp.insert(1, "group_name", group_name)
                agg_term_rows.append(tmp)

            if not norm_terms.empty:
                tmp = norm_terms.copy()
                tmp.insert(0, "group_type", group_type)
                tmp.insert(1, "group_name", group_name)
                agg_norm_rows.append(tmp)

            if not distinct_terms.empty:
                tmp = distinct_terms.copy()
                tmp.insert(0, "group_type", group_type)
                tmp.insert(1, "group_name", group_name)
                agg_distinct_rows.append(tmp)

    if agg_term_rows:
        pd.concat(agg_term_rows, ignore_index=True).to_csv(output_root / "aggregate_top_terms.csv", index=False)
    if agg_norm_rows:
        pd.concat(agg_norm_rows, ignore_index=True).to_csv(output_root / "aggregate_normalized_top_terms.csv", index=False)
    if agg_distinct_rows:
        pd.concat(agg_distinct_rows, ignore_index=True).to_csv(output_root / "aggregate_distinctive_terms.csv", index=False)

    representative = representative_utterances(all_df)
    if not representative.empty:
        representative.to_csv(output_root / "aggregate_representative_utterances.csv", index=False)

    embed_proj = embedding_projection(all_df, random_seed=CONFIG.random_seed)
    if not embed_proj.empty:
        embed_proj.to_csv(output_root / "aggregate_embedding_projection.csv", index=False)

    plots_dir = output_root / "plots"
    ensure_dir(plots_dir)

    for metric in METRICS_FOR_COMPARISON_PLOTS:
        if metric not in all_df.columns:
            continue
        model_metric = all_df.groupby("model_name")[metric].mean(numeric_only=True).sort_values(ascending=False)
        if not model_metric.empty:
            save_boxplot(all_df, "model_name", metric, f"Distribution by model - {metric}", plots_dir / f"aggregate_model_{metric}_boxplot.png", "Model", metric)

    for metric in METRICS_FOR_COMPARISON_PLOTS:
        if metric not in all_df.columns:
            continue
        role_metric = all_df.groupby("debate_role")[metric].mean(numeric_only=True).sort_values(ascending=False)
        if not role_metric.empty:
            save_barplot(role_metric, f"Aggregate role comparison - {metric}", plots_dir / f"aggregate_role_{metric}.png", metric, "Role")

    if not model_role_contrast.empty:
        for metric in METRICS_FOR_TRAJECTORY_PLOTS:
            sub = model_role_contrast[model_role_contrast["metric"] == metric]
            if sub.empty:
                continue
            gap_series = sub.set_index("model_name")["role_gap"].sort_values(ascending=False)
            save_barplot(gap_series, f"Role gap by model ({metric}) [proponent - opponent]", plots_dir / f"aggregate_role_gap_model_{metric}.png", "Gap", "Model")

    for metric in METRICS_FOR_TRAJECTORY_PLOTS:
        if metric not in all_df.columns:
            continue
        pivot = all_df.pivot_table(index="model_name", columns="debate_role", values=metric, aggfunc="mean")
        if not pivot.empty:
            save_heatmap(pivot, f"Model vs role heatmap - {metric}", plots_dir / f"aggregate_heatmap_model_role_{metric}.png", "Role", "Model")

    if "round" in all_df.columns:
        for metric in METRICS_FOR_TRAJECTORY_PLOTS:
            if metric not in all_df.columns:
                continue
            traj_df = all_df.groupby(["round", "model_name"], as_index=False)[metric].mean(numeric_only=True).sort_values(["model_name", "round"])
            if not traj_df.empty:
                pivot = traj_df.pivot(index="round", columns="model_name", values=metric).reset_index()
                save_lineplot(pivot, "round", [c for c in pivot.columns if c != "round"], f"Round trajectory by model - {metric}", plots_dir / f"aggregate_round_trajectory_model_{metric}.png", "Round", metric)



def process_single_run(run_dir: Path, topic_catalog: dict[str, dict], resources) -> tuple[pd.DataFrame, dict]:
    if not CONFIG.overwrite_existing:
        cached_df, cached_summary = load_cached_run_outputs(run_dir, CONFIG.output_analysis_dir)
        if cached_df is not None:
            print(f"Skipping recompute for {run_dir.name}: using cached outputs.")
            return cached_df, cached_summary or {"run_id": run_dir.name, "from_cache": True}

    run_meta = load_run_metadata(run_dir, topic_catalog)
    df, status = enrich_run_frame(run_dir, run_meta, resources)

    if status.get("skipped"):
        return pd.DataFrame(), {"run_id": run_dir.name, "skipped": True}

    summary = write_run_outputs(df, run_dir, run_meta | status, CONFIG.output_analysis_dir)
    return df, summary


def main(argv: list[str] | None = None) -> None:
    global CONFIG

    parser = argparse.ArgumentParser(description="LLM view analysis pipeline")
    parser.add_argument("--input-runs", type=str, default=str(CONFIG.input_runs_dir), help="Directory containing run_* folders")
    parser.add_argument("--output-analysis", type=str, default=str(CONFIG.output_analysis_dir), help="Directory where analysis outputs are written")
    parser.add_argument("--topics-path", type=str, default=str(CONFIG.topics_path), help="Path to topic catalog JSON")
    parser.add_argument("--max-runs", type=int, default=CONFIG.max_runs, help="Maximum number of runs to process")
    parser.add_argument("--overwrite-existing", action="store_true", help="Recompute existing per-run outputs")
    parser.add_argument("--use-transformer-emotion-secondary", action="store_true", help="Enable secondary transformer emotion model")
    parser.add_argument("--transformer-emotion-model", type=str, default=CONFIG.transformer_emotion_model, help="Transformer emotion model for secondary emotion features")
    args = parser.parse_args(argv)

    CONFIG = replace(
        CONFIG,
        input_runs_dir=Path(args.input_runs),
        output_analysis_dir=Path(args.output_analysis),
        topics_path=Path(args.topics_path),
        max_runs=args.max_runs,
        overwrite_existing=bool(args.overwrite_existing),
        use_transformer_emotion_secondary=bool(args.use_transformer_emotion_secondary),
        transformer_emotion_model=str(args.transformer_emotion_model),
    )

    configure_plot_style()

    random.seed(CONFIG.random_seed)
    np.random.seed(CONFIG.random_seed)

    if not CONFIG.input_runs_dir.exists():
        raise FileNotFoundError(f"Could not find input folder: {CONFIG.input_runs_dir}")

    ensure_dir(CONFIG.output_analysis_dir)
    topic_catalog = read_topics_catalog(CONFIG.topics_path)
    resources = load_nlp_resources(
        CONFIG.nrc_emotion_lexicon_path,
        CONFIG.use_transformer_emotion_secondary,
        CONFIG.transformer_emotion_model,
    )

    run_dirs = sorted([p for p in CONFIG.input_runs_dir.iterdir() if p.is_dir() and p.name.startswith("run_")])
    if not run_dirs:
        print("No run_* folders found.")
        return

    selected_runs = run_dirs if CONFIG.max_runs is None else run_dirs[: CONFIG.max_runs]

    print(f"Found {len(run_dirs)} run folders.")
    print(f"Processing {len(selected_runs)} runs:\n")
    for run_dir in selected_runs:
        print(f" - {run_dir.name}")
    print()

    all_frames: list[pd.DataFrame] = []
    run_summaries: list[dict] = []

    for run_dir in selected_runs:
        try:
            df, summary = process_single_run(run_dir, topic_catalog, resources)
            if not df.empty:
                all_frames.append(df)
            if summary and not summary.get("skipped"):
                run_summaries.append(summary)
        except Exception as exc:
            print(f"ERROR while processing {run_dir.name}: {exc}\n")

    if not all_frames:
        print("No debate rows were analyzed.")
        return

    all_df = pd.concat(all_frames, ignore_index=True)
    write_aggregate_outputs(all_df, run_summaries, CONFIG.output_analysis_dir)

    env_meta = {
        "python_version": sys.version,
        "platform": os.name,
        "random_seed": CONFIG.random_seed,
        "bootstrap_samples": CONFIG.bootstrap_samples,
        "analysis_schema_version": CONFIG.analysis_schema_version,
        "config": asdict(CONFIG),
        "libraries": {
            "spacy": __import__("spacy").__version__,
            "nltk": __import__("nltk").__version__,
            "sentence_transformers": getattr(sys.modules.get("sentence_transformers"), "__version__", None),
            "transformers": getattr(sys.modules.get("transformers"), "__version__", None),
            "wordcloud": getattr(sys.modules.get("wordcloud"), "__version__", None),
            "pandas": getattr(pd, "__version__", None),
            "numpy": getattr(np, "__version__", None),
            "matplotlib": getattr(__import__("matplotlib"), "__version__", None),
        },
        "spacy_pipeline": getattr(resources.nlp, "pipe_names", None) if resources.nlp is not None else None,
        "emotion_model_error": resources.emotion_load_error,
    }
    write_json(CONFIG.output_analysis_dir / "environment_metadata.json", env_meta)
    print(f"Analysis complete. Outputs written to: {CONFIG.output_analysis_dir}")


if __name__ == "__main__":
    main()
