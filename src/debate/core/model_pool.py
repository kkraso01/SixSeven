"""Model pool loader for batch simulations."""

from __future__ import annotations

from itertools import permutations, product
import json
import os
from pathlib import Path
from typing import Any

DEFAULT_MODEL_POOL_PATH = Path(__file__).resolve().parents[3] / "config" / "model_pool.json"


class ModelPoolError(ValueError):
    """Raised when model pool configuration is missing or malformed."""


def _resolve_model_pool_path() -> Path:
    env_path = os.getenv("DEBATE_MODEL_POOL_FILE")
    return Path(env_path).expanduser() if env_path else DEFAULT_MODEL_POOL_PATH


def load_model_pool() -> dict[str, Any]:
    """Load model pool JSON from disk."""
    pool_path = _resolve_model_pool_path()
    with pool_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ModelPoolError(f"Model pool JSON must contain an object: {pool_path}")
    return data


def _build_model_registry(pool: dict[str, Any]) -> dict[str, str]:
    raw_models = pool.get("models", [])
    if not isinstance(raw_models, list):
        raise ModelPoolError("model_pool.models must be a list")

    registry: dict[str, str] = {}
    for idx, item in enumerate(raw_models, start=1):
        if not isinstance(item, dict):
            raise ModelPoolError(f"model_pool.models[{idx}] must be an object")

        name = item.get("name")
        api_mode = item.get("api_mode")
        if not isinstance(name, str) or not name.strip():
            raise ModelPoolError(f"model_pool.models[{idx}].name must be a non-empty string")
        if not isinstance(api_mode, str) or not api_mode.strip():
            raise ModelPoolError(f"model_pool.models[{idx}].api_mode must be a non-empty string")

        normalized_name = name.strip()
        normalized_mode = api_mode.strip().lower()
        if normalized_name in registry:
            raise ModelPoolError(f"Duplicate model name in pool: {normalized_name}")
        registry[normalized_name] = normalized_mode

    if not registry:
        raise ModelPoolError("model_pool.models must contain at least one model")

    return registry


def _infer_combo_api_mode(combo: tuple[str, str, str], registry: dict[str, str]) -> str:
    modes = {registry[model] for model in combo}
    return "gemini" if "gemini" in modes else "openai"


def get_preferred_gemini_model(default: str = "gemini-3-flash-preview") -> str:
    """Return a Gemini model from the pool, falling back to a safe default."""
    pool = load_model_pool()
    registry = _build_model_registry(pool)
    for model_name, api_mode in registry.items():
        if api_mode == "gemini":
            return model_name
    return default


def get_default_role_models() -> dict[str, str]:
    """Return fallback role models inferred from the flat model list.

    Preference:
    1) First three non-Gemini models
    2) First three models in the list
    """
    pool = load_model_pool()
    registry = _build_model_registry(pool)

    ordered_models = list(registry.keys())
    non_gemini = [name for name, mode in registry.items() if mode != "gemini"]
    chosen = non_gemini if len(non_gemini) >= 3 else ordered_models

    if len(chosen) < 3:
        raise ModelPoolError("At least 3 models are required to infer default role models")

    return {
        "moderator": chosen[0],
        "conspiracy": chosen[1],
        "scientific": chosen[2],
    }


def get_batch_model_configs(batch_name: str) -> list[dict[str, str]]:
    """Generate batch model permutations from the flat model list.

    Rules:
<<<<<<< HEAD
    - ollama batch: only non-Gemini models, no repeated model in one combo
=======
        - ollama batch: only non-Gemini models
            - with exactly 2 models: allow repeats to generate all role assignments
            - with 3+ models: no repeated model in one combo
>>>>>>> dev
    - gemini batch: all models, repeated models allowed, but combo must include at least one Gemini
    """
    pool = load_model_pool()
    registry = _build_model_registry(pool)

    if batch_name == "ollama":
        candidates = [name for name, mode in registry.items() if mode != "gemini"]
<<<<<<< HEAD
        if len(candidates) < 3:
            raise ModelPoolError("Ollama batch requires at least 3 non-Gemini models")
        combos = list(permutations(candidates, 3))
=======
        if len(candidates) < 2:
            raise ModelPoolError("Ollama batch requires at least 2 non-Gemini models")

        if len(candidates) == 2:
            combos = list(product(candidates, repeat=3))
        else:
            combos = list(permutations(candidates, 3))
>>>>>>> dev
    elif batch_name == "gemini":
        candidates = list(registry.keys())
        if len(candidates) < 1:
            raise ModelPoolError("Gemini batch requires at least one model")

        combos = [
            combo
            for combo in product(candidates, repeat=3)
            if any(registry[model] == "gemini" for model in combo)
        ]
        if not combos:
            raise ModelPoolError("Gemini batch requires at least one model with api_mode='gemini'")
    else:
        raise ModelPoolError(f"Unsupported batch name: {batch_name}")

    return [
        {
            "name": f"MOD={moderator}_CA={conspiracy}_SA={scientific}",
            "moderator": moderator,
            "conspiracy": conspiracy,
            "scientific": scientific,
            "api_mode": _infer_combo_api_mode((moderator, conspiracy, scientific), registry),
        }
        for moderator, conspiracy, scientific in combos
    ]
