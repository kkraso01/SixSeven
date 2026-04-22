from __future__ import annotations

import re
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from .config import ROLE_OPPONENT_MAP
from debate.analysis.winner_inference import infer_winner_from_final_report as shared_infer_winner_from_final_report


def infer_winner_from_final_report(final_report: dict) -> dict:
    return shared_infer_winner_from_final_report(
        final_report,
        use_explicit_winner_fields=True,
        use_stance_trajectory_fallback=False,
    )


def lexical_count(tokens: list[str], lexicon: set[str]) -> int:
    return sum(token in lexicon for token in tokens)


def jaccard_similarity(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    denom = len(sa | sb)
    return 0.0 if denom == 0 else len(sa & sb) / denom


def lexical_overlap(a: list[str], b: list[str], stop_words: set[str]) -> float:
    left = [x for x in a if x not in stop_words]
    right = [x for x in b if x not in stop_words]
    return jaccard_similarity(left, right)


def stance_proxy_score(tokens: list[str], role: str, pro_words: set[str], con_words: set[str]) -> float:
    pro_score = lexical_count(tokens, pro_words)
    con_score = lexical_count(tokens, con_words)
    raw = float(pro_score - con_score)
    if role == "opponent":
        raw = -raw
    return raw


def compute_stage_labels(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype="object")
    n = len(df)
    if n == 1:
        return pd.Series(["opening"], index=df.index)
    positions = np.linspace(0, 1, n, endpoint=False)
    labels: list[str] = []
    for pos in positions:
        if pos < 1 / 3:
            labels.append("opening")
        elif pos < 2 / 3:
            labels.append("middle")
        else:
            labels.append("closing")
    return pd.Series(labels, index=df.index)


def cosine_with_prev(vec: np.ndarray | None, prev_vec: np.ndarray | None) -> float:
    if vec is None or prev_vec is None:
        return np.nan
    try:
        return float(cosine_similarity([vec], [prev_vec])[0, 0])
    except Exception:
        return np.nan


def compute_interaction_features(df: pd.DataFrame, stop_words: set[str]) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.sort_values("turn_index").copy()

    prev_tokens: list[str] | None = None
    prev_vec: np.ndarray | None = None
    last_turn_by_role: dict[str, tuple[list[str], np.ndarray | None]] = {}
    prev_nonself_tokens: list[str] | None = None
    prev_nonself_vec: np.ndarray | None = None
    previous_role: str | None = None

    lexical_prev = []
    lexical_prev_other = []
    semantic_prev = []
    semantic_prev_other = []
    lexical_prev_opponent = []
    semantic_prev_opponent = []

    for _, row in out.iterrows():
        role = str(row["speaker_role"])
        lemmas = row.get("lemmas_list") or []
        vec = row.get("embedding_vector_obj")

        current_prev_nonself_tokens = prev_nonself_tokens
        current_prev_nonself_vec = prev_nonself_vec
        if prev_tokens is not None and previous_role is not None and previous_role != role:
            current_prev_nonself_tokens = prev_tokens
            current_prev_nonself_vec = prev_vec

        lexical_prev.append(lexical_overlap(lemmas, prev_tokens or [], stop_words))
        semantic_prev.append(cosine_with_prev(vec, prev_vec))
        lexical_prev_other.append(lexical_overlap(lemmas, current_prev_nonself_tokens or [], stop_words))
        semantic_prev_other.append(cosine_with_prev(vec, current_prev_nonself_vec))

        opponent_role = ROLE_OPPONENT_MAP.get(role)
        opp_tokens, opp_vec = last_turn_by_role.get(opponent_role, ([], None))
        lexical_prev_opponent.append(lexical_overlap(lemmas, opp_tokens, stop_words))
        semantic_prev_opponent.append(cosine_with_prev(vec, opp_vec))

        prev_nonself_tokens = current_prev_nonself_tokens
        prev_nonself_vec = current_prev_nonself_vec
        prev_tokens = lemmas
        prev_vec = vec
        last_turn_by_role[role] = (lemmas, vec)
        previous_role = role

    out["lexical_overlap_prev_turn"] = lexical_prev
    out["lexical_overlap_prev_other_turn"] = lexical_prev_other
    out["semantic_similarity_prev_turn"] = semantic_prev
    out["semantic_similarity_prev_other_turn"] = semantic_prev_other
    out["lexical_overlap_prev_opponent_turn"] = lexical_prev_opponent
    out["semantic_similarity_prev_opponent_turn"] = semantic_prev_opponent

    # Novelty proxy: higher value means less overlap with the previous response context.
    out["novelty_vs_prev_turn_lexical"] = 1.0 - out["lexical_overlap_prev_turn"].clip(lower=0.0, upper=1.0)
    out["novelty_vs_prev_turn_semantic"] = 1.0 - out["semantic_similarity_prev_turn"].clip(lower=-1.0, upper=1.0)
    out["novelty_vs_prev_opponent_turn_lexical"] = 1.0 - out["lexical_overlap_prev_opponent_turn"].clip(lower=0.0, upper=1.0)
    out["novelty_vs_prev_opponent_turn_semantic"] = 1.0 - out["semantic_similarity_prev_opponent_turn"].clip(lower=-1.0, upper=1.0)
    return out.sort_index()


def build_stance_change_events(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty or "stance_proxy_score" not in df.columns:
        return pd.DataFrame()
    needed = ["run_id", "turn_index", "stance_proxy_score"] + group_cols
    sub = df[needed].dropna(subset=["stance_proxy_score"]).sort_values(["run_id", "turn_index"]).copy()
    if sub.empty:
        return pd.DataFrame()

    rows = []
    for keys, group in sub.groupby(["run_id"] + group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        run_id = keys[0]
        grouped_values = keys[1:]
        base = {"run_id": run_id, **{col: val for col, val in zip(group_cols, grouped_values)}}
        initial = float(group.iloc[0]["stance_proxy_score"])
        final = float(group.iloc[-1]["stance_proxy_score"])
        delta = final - initial
        initial_sign = int(np.sign(initial))
        final_sign = int(np.sign(final))
        has_flip = int(initial_sign != 0 and final_sign != 0 and initial_sign != final_sign)
        rows.append(
            {
                **base,
                "initial_stance_proxy": initial,
                "final_stance_proxy": final,
                "stance_proxy_delta": delta,
                "initial_stance_sign": initial_sign,
                "final_stance_sign": final_sign,
                "stance_flip": has_flip,
                "turns_in_group": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def summarise_stance_change(events_df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if events_df.empty:
        return pd.DataFrame()
    out = (
        events_df.groupby(group_cols, as_index=False)
        .agg(
            debates=("run_id", "nunique"),
            stance_flip_count=("stance_flip", "sum"),
            avg_abs_stance_shift=("stance_proxy_delta", lambda s: float(np.mean(np.abs(s)))),
            avg_signed_stance_shift=("stance_proxy_delta", "mean"),
        )
    )
    out["stance_flip_rate"] = out.apply(
        lambda row: 0.0 if float(row["debates"]) == 0 else float(row["stance_flip_count"]) / float(row["debates"]),
        axis=1,
    )
    return out


def build_turning_point_events(
    df: pd.DataFrame,
    group_cols: list[str],
    delta_col: str = "confidence_delta_within_role",
    threshold: float = 5.0,
) -> pd.DataFrame:
    if df.empty or delta_col not in df.columns:
        return pd.DataFrame()
    needed = ["run_id", "turn_index", delta_col] + group_cols
    sub = df[needed].dropna(subset=[delta_col]).copy()
    if sub.empty:
        return pd.DataFrame()
    sub["abs_delta"] = sub[delta_col].abs()
    sub["is_turning_point"] = (sub["abs_delta"] >= threshold).astype(int)
    return sub


def summarise_turning_points(
    events_df: pd.DataFrame,
    group_cols: list[str],
    delta_col: str = "confidence_delta_within_role",
) -> pd.DataFrame:
    if events_df.empty or "is_turning_point" not in events_df.columns:
        return pd.DataFrame()
    out = (
        events_df.groupby(group_cols, as_index=False)
        .agg(
            observed_turns=("turn_index", "count"),
            turning_point_count=("is_turning_point", "sum"),
            max_abs_delta=("abs_delta", "max"),
            avg_abs_delta=("abs_delta", "mean"),
            avg_signed_delta=(delta_col, "mean"),
        )
    )
    out["turning_point_rate"] = out.apply(
        lambda row: 0.0 if float(row["observed_turns"]) == 0 else float(row["turning_point_count"]) / float(row["observed_turns"]),
        axis=1,
    )
    return out


def compute_role_alignment(
    predicted_stance_label: str,
    support_score: float,
    oppose_score: float,
    speaker_role: str,
) -> tuple[str, float]:
    """
    Compute role alignment label and score based on predicted stance and assigned role.
    
    Args:
        predicted_stance_label: "support", "oppose", or "neutral"
        support_score: Model score for support
        oppose_score: Model score for oppose
        speaker_role: "proponent" or "opponent"
    
    Returns:
        (role_alignment_label, role_alignment_score)
        - role_alignment_label: "aligned", "neutral", or "misaligned"
        - role_alignment_score: positive (aligned), zero (neutral), negative (misaligned)
    """
    if predicted_stance_label == "neutral":
        return "neutral", 0.0
    
    if speaker_role == "proponent":
        # For proponents: support claim = aligned, oppose claim = misaligned
        if predicted_stance_label == "support":
            alignment = "aligned"
            score = support_score - oppose_score
        else:  # oppose
            alignment = "misaligned"
            score = oppose_score - support_score  # negative because misaligned
        return alignment, float(score)
    
    elif speaker_role == "opponent":
        # For opponents: oppose claim = aligned, support claim = misaligned
        if predicted_stance_label == "oppose":
            alignment = "aligned"
            score = oppose_score - support_score
        else:  # support
            alignment = "misaligned"
            score = support_score - oppose_score  # negative because misaligned
        return alignment, float(score)
    
    # Fallback for unknown role
    return "neutral", 0.0


def build_role_alignment_events(
    df: pd.DataFrame,
    group_cols: list[str],
) -> pd.DataFrame:
    """
    Build events where role alignment changes or flips within a speaker's role.
    
    Args:
        df: Dataframe with role_alignment_label and role_alignment_score columns
        group_cols: Columns to group by (typically ["run_id", "speaker_role"])
    
    Returns:
        DataFrame with alignment change events
    """
    if df.empty or "role_alignment_score" not in df.columns:
        return pd.DataFrame()
    
    needed = ["run_id", "turn_index", "role_alignment_label", "role_alignment_score"] + group_cols
    if not all(col in df.columns for col in needed):
        return pd.DataFrame()
    
    sub = df[needed].copy()
    rows = []
    
    for key, group in sub.groupby(group_cols):
        group = group.sort_values("turn_index").reset_index(drop=True)
        if len(group) < 2:
            continue
        
        for i in range(1, len(group)):
            prev_label = group.iloc[i-1]["role_alignment_label"]
            curr_label = group.iloc[i]["role_alignment_label"]
            prev_score = group.iloc[i-1]["role_alignment_score"]
            curr_score = group.iloc[i]["role_alignment_score"]
            
            # Detect alignment changes
            if prev_label != curr_label:
                change_type = f"{prev_label}→{curr_label}"
                score_change = curr_score - prev_score
                rows.append({
                    "run_id": group.iloc[i]["run_id"],
                    "turn_index": group.iloc[i]["turn_index"],
                    "alignment_change_type": change_type,
                    "role_alignment_score_delta": score_change,
                    "is_misalignment_event": 1 if curr_label == "misaligned" else 0,
                    **{col: group.iloc[i][col] for col in group_cols if col != "run_id"},
                })
    
    return pd.DataFrame(rows) if rows else pd.DataFrame()
