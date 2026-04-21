from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ROLE_ALIASES


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json_if_exists(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return None


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def write_json(path: Path, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False, default=str)


def save_text_report(path: Path, content: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(content)


def canonical_role(value: Any) -> str:
    role = str(value).strip().lower().replace("-", " ").replace("_", " ")
    role = re.sub(r"\s+", " ", role)
    compact = role.replace(" ", "_")
    return ROLE_ALIASES.get(compact, ROLE_ALIASES.get(role, compact or "unknown"))


def normalize_name(value: Any) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value)).strip("_") or "unknown"


def read_topics_catalog(path: Path) -> dict[str, dict]:
    data = read_json_if_exists(path) or []
    return {str(item.get("id")): item for item in data if item.get("id") is not None}


def parse_json_list(value: Any) -> list[Any] | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except Exception:
        return None
    return parsed if isinstance(parsed, list) else None


def rehydrate_cached_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["tokens_list", "lemmas_list", "pos_tags_list"]:
        if col in out.columns:
            out[col] = out[col].apply(parse_json_list)
    if "embedding_vector_obj" in out.columns:
        out["embedding_vector_obj"] = out["embedding_vector_obj"].apply(parse_json_list)
        out["embedding_vector_obj"] = out["embedding_vector_obj"].apply(
            lambda x: np.array(x, dtype=float) if isinstance(x, list) and x else None
        )
    return out


def persist_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["tokens_list", "lemmas_list", "pos_tags_list", "embedding_vector_obj"]:
        if col in out.columns:
            out[col] = out[col].apply(
                lambda x: json.dumps(x.tolist()) if isinstance(x, np.ndarray)
                else json.dumps(x) if isinstance(x, list)
                else None
            )
    return out


def save_frames(output_dir: Path, frames: dict[str, pd.DataFrame]) -> None:
    for name, frame in frames.items():
        if frame is not None and not frame.empty:
            frame.to_csv(output_dir / name, index=False)
